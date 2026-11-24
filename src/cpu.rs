//! The CPU backend: the format-9 hybrid tree forward pass of upstream
//! `packages/training/src/tree-model.js` (`treeProbabilities`,
//! `hybridLeaves`, `descend`, `classify`), the JS reference the GPU
//! shader is checked against.
//!
//! The reference stores every activation in a `Float32Array` and computes
//! in JS numbers (`f64`). This port does the same, in the same order of
//! operations: sums accumulate in `f64` in the reference's order and are
//! rounded to `f32` where the reference stores them. The loops are
//! vectorized across output units (never across a sum) and parallel across
//! parts, so the speed comes without reordering any arithmetic.

use crate::math::{exp, sigmoid, tanh};
use crate::model::{weights, Weights, AUX, CLS, CLS_IN, H, JOINED, OUT};
use crate::prepare::{feature_indices, Prepared};

/// Runs `f(index, chunk)` over `chunk`-sized pieces of `out`, in parallel
/// where threads are available.
fn for_each_chunk<T: Send>(out: &mut [T], chunk: usize, f: impl Fn(usize, &mut [T]) + Sync + Send) {
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    {
        use rayon::prelude::*;
        if out.len() / chunk.max(1) >= 256 {
            out.par_chunks_mut(chunk)
                .enumerate()
                .with_min_len(64)
                .for_each(|(index, piece)| f(index, piece));
            return;
        }
    }
    for (index, piece) in out.chunks_mut(chunk).enumerate() {
        f(index, piece);
    }
}

/// One label byte per part: the class in the low four bits, the selected
/// class's probability in sixteenths (rounded down, at most 15) in the
/// high four. Whitespace is plain at level 15, as in the GPU runtime.
pub fn labels(prepared: &Prepared) -> Vec<u8> {
    let count = prepared.len();
    if count == 0 {
        return Vec::new();
    }
    let w = weights();
    let raw = embed(w, prepared);
    let leaves = hybrid_leaves(w, prepared, &raw);
    drop(raw);
    let context = tree_context(w, leaves.clone(), count);
    let mut labels = vec![0u8; count];
    for_each_chunk(&mut labels, 1, |index, label| {
        label[0] = if prepared.is_space(index) {
            15 << 4
        } else {
            classify(
                w,
                &leaves[index * H..(index + 1) * H],
                &context[index * H..(index + 1) * H],
            )
        };
    });
    labels
}

/// Sums each part's sparse feature rows (`Float32Array` accumulation).
fn embed(w: &Weights, prepared: &Prepared) -> Vec<f32> {
    let mut raw = vec![0f32; prepared.len() * H];
    for_each_chunk(&mut raw, H, |index, out| {
        let mut rows = [0u16; 17];
        let n = feature_indices(&prepared.data, index, &mut rows);
        for &row in &rows[..n] {
            let embedding = &w.embedding[row as usize * H..(row as usize + 1) * H];
            for h in 0..H {
                out[h] += embedding[h];
            }
        }
    });
    raw
}

/// `hybridLeaves`: local window, gated bidirectional scans and their mix.
fn hybrid_leaves(w: &Weights, prepared: &Prepared, raw: &[f32]) -> Vec<f32> {
    let count = prepared.len();
    // Nearest non-whitespace part before and after each part.
    let mut previous = vec![usize::MAX; count];
    let mut next = vec![usize::MAX; count];
    let mut last = usize::MAX;
    for i in 0..count {
        previous[i] = last;
        if !prepared.is_space(i) {
            last = i;
        }
    }
    last = usize::MAX;
    for i in (0..count).rev() {
        next[i] = last;
        if !prepared.is_space(i) {
            last = i;
        }
    }
    let at = |index: usize, h: usize| -> f64 {
        if index < count { f64::from(raw[index * H + h]) } else { 0.0 }
    };

    let mut local = vec![0f32; count * H];
    for_each_chunk(&mut local, H, |i, out| {
        for h in 0..H {
            let mut value = w.leaf_bias[h];
            for offset in 0..5usize {
                let index = (i + offset).wrapping_sub(2);
                value += at(index, h) * w.local_offset[offset * H + h];
            }
            value += at(previous[i], h) * w.local_nonspace[h];
            value += at(next[i], h) * w.local_nonspace[H + h];
            out[h] = tanh(value) as f32;
        }
    });

    // Gates and values: [part][gate (32) | value (32)].
    let mut gates_values = vec![0f32; count * H * 2];
    for_each_chunk(&mut gates_values, H * 2, |i, out| {
        let input = &local[i * H..(i + 1) * H];
        let mut gate = [0f64; H];
        let mut value = [0f64; H];
        gate.copy_from_slice(&w.state_gate_bias);
        value.copy_from_slice(&w.state_input_bias);
        for k in 0..H {
            let x = f64::from(input[k]);
            let gate_row = &w.state_gate_t[k * H..(k + 1) * H];
            let value_row = &w.state_input_t[k * H..(k + 1) * H];
            for h in 0..H {
                gate[h] += x * gate_row[h];
                value[h] += x * value_row[h];
            }
        }
        for h in 0..H {
            out[h] = sigmoid(gate[h]) as f32;
            out[H + h] = tanh(value[h]) as f32;
        }
    });

    let scan = |reverse: bool| -> Vec<f32> {
        let mut result = vec![0f32; count * H];
        let mut state = [0f32; H];
        for step in 0..count {
            let i = if reverse { count - 1 - step } else { step };
            let gv = &gates_values[i * H * 2..(i + 1) * H * 2];
            for h in 0..H {
                let gate = f64::from(gv[h]);
                state[h] = (gate * f64::from(state[h]) + (1.0 - gate) * f64::from(gv[H + h])) as f32;
            }
            result[i * H..(i + 1) * H].copy_from_slice(&state);
        }
        result
    };
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    let (forward, backward) = rayon::join(|| scan(false), || scan(true));
    #[cfg(not(all(feature = "parallel", not(target_arch = "wasm32"))))]
    let (forward, backward) = (scan(false), scan(true));
    drop(gates_values);

    let mut leaves = vec![0f32; count * H];
    for_each_chunk(&mut leaves, H, |i, out| {
        let mut sum = [0f64; H];
        sum.copy_from_slice(&w.state_mix_bias);
        let mut accumulate = |inputs: &[f32], first: usize| {
            for (k, &x) in inputs.iter().enumerate() {
                let x = f64::from(x);
                let row = &w.state_mix_t[(first + k) * H..(first + k + 1) * H];
                for h in 0..H {
                    sum[h] += x * row[h];
                }
            }
        };
        accumulate(&forward[i * H..(i + 1) * H], 0);
        accumulate(&backward[i * H..(i + 1) * H], H);
        for h in 0..H {
            out[h] = tanh(f64::from(local[i * H + h]) + sum[h]) as f32;
        }
    });
    leaves
}

/// The scale-aware butterfly tree: merges leaves bottom-up, then sends
/// whole-source context top-down. Returns each leaf's returned context.
fn tree_context(w: &Weights, leaves: Vec<f32>, count: usize) -> Vec<f32> {
    let scale_buckets = w.merge_bias.len() / H;
    let max_shift_log = (H.ilog2() as usize).saturating_sub(1);
    let mut levels: Vec<Vec<f32>> = vec![leaves];
    let mut counts = vec![count];
    let mut depth = 0usize;
    while *counts.last().unwrap() > 1 {
        let children = levels.last().unwrap();
        let child_count = *counts.last().unwrap();
        let parent_count = child_count.div_ceil(2);
        let scale = depth.min(scale_buckets - 1) * H;
        let shift = 1usize << depth.min(max_shift_log);
        let mut parents = vec![0f32; parent_count * H];
        for_each_chunk(&mut parents, H, |p, out| {
            let left = &children[2 * p * H..(2 * p + 1) * H];
            if 2 * p + 1 >= child_count {
                out.copy_from_slice(left);
                return;
            }
            let right = &children[(2 * p + 1) * H..(2 * p + 2) * H];
            for h in 0..H {
                let partner = h ^ shift;
                let value = f64::from(left[h]);
                let mixed = tanh(
                    value * w.merge_own_left[scale + h]
                        + f64::from(right[h]) * w.merge_own_right[scale + h]
                        + f64::from(left[partner]) * w.merge_cross_left[scale + h]
                        + f64::from(right[partner]) * w.merge_cross_right[scale + h]
                        + w.merge_bias[scale + h],
                );
                let boundary = if h < H / 2 { value } else { f64::from(right[h]) };
                out[h] = ((mixed + boundary) * 0.5) as f32;
            }
        });
        levels.push(parents);
        counts.push(parent_count);
        depth += 1;
    }

    let mut down = levels.pop().unwrap();
    for depth in (0..levels.len()).rev() {
        let children = &levels[depth];
        let child_count = counts[depth];
        let scale = depth.min(scale_buckets - 1) * H;
        let shift = 1usize << depth.min(max_shift_log);
        let mut next = vec![0f32; child_count * H];
        // Pairs of children, written together from their parent.
        for_each_chunk(&mut next, 2 * H, |p, out| {
            let parent = &down[p * H..(p + 1) * H];
            let left = &children[2 * p * H..(2 * p + 1) * H];
            if 2 * p + 1 >= child_count {
                out[..H].copy_from_slice(parent);
                return;
            }
            let right = &children[(2 * p + 1) * H..(2 * p + 2) * H];
            let (out_left, out_right) = out.split_at_mut(H);
            descend(w, parent, left, right, scale, shift, false, out_left);
            descend(w, parent, right, left, scale, shift, true, out_right);
        });
        down = next;
    }
    down
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn descend(
    w: &Weights,
    parent: &[f32],
    own: &[f32],
    sibling: &[f32],
    scale: usize,
    shift: usize,
    right: bool,
    out: &mut [f32],
) {
    let bias = if right { &w.down_right_bias } else { &w.down_left_bias };
    for h in 0..H {
        let index = scale + h;
        let partner = h ^ shift;
        let value = f64::from(parent[h]);
        let mixed = tanh(
            value * w.down_own_parent[index]
                + f64::from(own[h]) * w.down_own_self[index]
                + f64::from(sibling[h]) * w.down_own_sibling[index]
                + f64::from(parent[partner]) * w.down_cross_parent[index]
                + f64::from(own[partner]) * w.down_cross_self[index]
                + f64::from(sibling[partner]) * w.down_cross_sibling[index]
                + bias[index],
        );
        let skip = w.down_skip[index];
        out[h] = (value * skip + mixed * (1.0 - skip)) as f32;
    }
}

/// `classify` and the label byte: auxiliary states, classifier, softmax.
fn classify(w: &Weights, leaf: &[f32], context: &[f32]) -> u8 {
    let mut joined = [0f64; CLS_IN];
    for h in 0..H {
        joined[h] = f64::from(leaf[h]);
        joined[H + h] = f64::from(context[h]);
    }
    let mut auxiliary = [0f64; AUX];
    auxiliary.copy_from_slice(&w.auxiliary_bias);
    for k in 0..JOINED {
        let x = joined[k];
        let row = &w.auxiliary_t[k * AUX..(k + 1) * AUX];
        for a in 0..AUX {
            auxiliary[a] += row[a] * x;
        }
    }
    for a in 0..AUX {
        joined[JOINED + a] = f64::from(sigmoid(auxiliary[a]) as f32);
    }

    let mut classifier = [0f64; CLS];
    classifier.copy_from_slice(&w.classifier_bias);
    for k in 0..CLS_IN {
        let x = joined[k];
        let row = &w.classifier_t[k * CLS..(k + 1) * CLS];
        for c in 0..CLS {
            classifier[c] += row[c] * x;
        }
    }
    for value in &mut classifier {
        *value = f64::from(tanh(*value) as f32);
    }

    let mut sums = [0f64; OUT];
    sums.copy_from_slice(&w.output_bias);
    for c in 0..CLS {
        let x = classifier[c];
        let row = &w.output_t[c * OUT..(c + 1) * OUT];
        for o in 0..OUT {
            sums[o] += row[o] * x;
        }
    }
    let mut maximum = f64::NEG_INFINITY;
    for &sum in &sums {
        maximum = maximum.max(sum);
    }
    // logits[o] is stored as f32; the denominator adds the unrounded exp.
    let mut stored = [0f32; OUT];
    let mut denominator = 0f64;
    for o in 0..OUT {
        let e = exp(f64::from(sums[o] as f32) - maximum);
        stored[o] = e as f32;
        denominator += e;
    }
    let mut probabilities = [0f32; OUT];
    for o in 0..OUT {
        probabilities[o] = (f64::from(stored[o]) / denominator) as f32;
    }
    let mut predicted = 0;
    for o in 1..OUT {
        if probabilities[o] > probabilities[predicted] {
            predicted = o;
        }
    }
    let level = (16.0 * f64::from(probabilities[predicted])).floor().min(15.0) as u8;
    predicted as u8 | (level << 4)
}
