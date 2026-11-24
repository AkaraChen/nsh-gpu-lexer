//! The promoted checkpoint, dequantized once for the CPU forward pass.
//!
//! Each weight is `code * scale` rounded to `f32`, exactly as upstream
//! `dequantizeTensors` fills its `Float32Array`. The CPU pass then reads
//! them as `f64` (lossless) and keeps matrices transposed so the inner
//! loops run across output units without changing any sum's order.

use std::sync::OnceLock;

use crate::generated::model::{self as m, TensorSpec};

pub(crate) const H: usize = m::HIDDEN;
pub(crate) const AUX: usize = m::AUXILIARY;
pub(crate) const CLS: usize = m::CLASSIFIER;
pub(crate) const OUT: usize = m::CLASSES;
pub(crate) const JOINED: usize = H * 2;
pub(crate) const CLS_IN: usize = H * 2 + AUX;

/// The weights in the shapes the CPU pass reads.
pub(crate) struct Weights {
    pub embedding: Vec<f32>,
    pub leaf_bias: Vec<f64>,
    pub local_offset: Vec<f64>,
    pub local_nonspace: Vec<f64>,
    /// `[input][unit]`.
    pub state_gate_t: Vec<f64>,
    pub state_gate_bias: Vec<f64>,
    pub state_input_t: Vec<f64>,
    pub state_input_bias: Vec<f64>,
    /// `[input (64)][unit]`.
    pub state_mix_t: Vec<f64>,
    pub state_mix_bias: Vec<f64>,
    pub merge_own_left: Vec<f64>,
    pub merge_own_right: Vec<f64>,
    pub merge_cross_left: Vec<f64>,
    pub merge_cross_right: Vec<f64>,
    pub merge_bias: Vec<f64>,
    pub down_own_parent: Vec<f64>,
    pub down_own_self: Vec<f64>,
    pub down_own_sibling: Vec<f64>,
    pub down_cross_parent: Vec<f64>,
    pub down_cross_self: Vec<f64>,
    pub down_cross_sibling: Vec<f64>,
    /// `1 / (1 + exp(-downSkip))`, as `descend` computes it each time.
    pub down_skip: Vec<f64>,
    pub down_left_bias: Vec<f64>,
    pub down_right_bias: Vec<f64>,
    /// `[input (64)][auxiliary]`.
    pub auxiliary_t: Vec<f64>,
    pub auxiliary_bias: Vec<f64>,
    /// `[input (80)][unit]`.
    pub classifier_t: Vec<f64>,
    pub classifier_bias: Vec<f64>,
    /// `[unit][class]`.
    pub output_t: Vec<f64>,
    pub output_bias: Vec<f64>,
}

/// Unpacks one signed code, as `dequantizeTensors` reads them.
fn code(data: &[u8], index: usize) -> i32 {
    let bits = m::WEIGHT_BITS as usize;
    let mask = (1u32 << bits) - 1;
    let sign = 1u32 << (bits - 1);
    let bit_offset = index * bits;
    let byte = bit_offset >> 3;
    let shift = bit_offset & 7;
    let low = u32::from(data[byte]);
    let high = u32::from(data.get(byte + 1).copied().unwrap_or(0));
    let encoded = ((low | (high << 8)) >> shift) & mask;
    if encoded & sign != 0 {
        encoded as i32 - mask as i32 - 1
    } else {
        encoded as i32
    }
}

/// A tensor's values, rounded to `f32` as the reference stores them.
pub(crate) fn tensor(spec: TensorSpec) -> Vec<f32> {
    (0..spec.length)
        .map(|index| (f64::from(code(m::WEIGHTS_INT6, spec.offset + index)) * spec.scale) as f32)
        .collect()
}

fn wide(values: &[f32]) -> Vec<f64> {
    values.iter().map(|&v| f64::from(v)).collect()
}

/// `[rows][columns]` to `[columns][rows]`.
fn transpose(values: &[f32], rows: usize, columns: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows * columns];
    for row in 0..rows {
        for column in 0..columns {
            out[column * rows + row] = f64::from(values[row * columns + column]);
        }
    }
    out
}

impl Weights {
    fn load() -> Weights {
        let t = |spec: TensorSpec| tensor(spec);
        let w = |spec: TensorSpec| wide(&tensor(spec));
        Weights {
            embedding: t(m::FEATURE_EMBEDDING),
            leaf_bias: w(m::LEAF_BIAS),
            local_offset: w(m::LOCAL_OFFSET_SCALE),
            local_nonspace: w(m::LOCAL_NONSPACE_SCALE),
            state_gate_t: transpose(&t(m::STATE_GATE), H, H),
            state_gate_bias: w(m::STATE_GATE_BIAS),
            state_input_t: transpose(&t(m::STATE_INPUT), H, H),
            state_input_bias: w(m::STATE_INPUT_BIAS),
            state_mix_t: transpose(&t(m::STATE_MIX), H, H * 2),
            state_mix_bias: w(m::STATE_MIX_BIAS),
            merge_own_left: w(m::MERGE_OWN_LEFT),
            merge_own_right: w(m::MERGE_OWN_RIGHT),
            merge_cross_left: w(m::MERGE_CROSS_LEFT),
            merge_cross_right: w(m::MERGE_CROSS_RIGHT),
            merge_bias: w(m::MERGE_BIAS),
            down_own_parent: w(m::DOWN_OWN_PARENT),
            down_own_self: w(m::DOWN_OWN_SELF),
            down_own_sibling: w(m::DOWN_OWN_SIBLING),
            down_cross_parent: w(m::DOWN_CROSS_PARENT),
            down_cross_self: w(m::DOWN_CROSS_SELF),
            down_cross_sibling: w(m::DOWN_CROSS_SIBLING),
            down_skip: tensor(m::DOWN_SKIP)
                .iter()
                .map(|&v| crate::math::sigmoid(f64::from(v)))
                .collect(),
            down_left_bias: w(m::DOWN_LEFT_BIAS),
            down_right_bias: w(m::DOWN_RIGHT_BIAS),
            auxiliary_t: transpose(&t(m::AUXILIARY_OUTPUT), AUX, JOINED),
            auxiliary_bias: w(m::AUXILIARY_BIAS),
            classifier_t: transpose(&t(m::CLASSIFIER_INPUT), CLS, CLS_IN),
            classifier_bias: w(m::CLASSIFIER_BIAS),
            output_t: transpose(&t(m::OUTPUT), OUT, CLS),
            output_bias: w(m::OUTPUT_BIAS),
        }
    }
}

/// The dequantized weights, built on first use (about 1 ms).
pub(crate) fn weights() -> &'static Weights {
    static WEIGHTS: OnceLock<Weights> = OnceLock::new();
    WEIGHTS.get_or_init(Weights::load)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_table_is_contiguous_and_complete() {
        let mut offset = 0;
        for spec in m::TENSORS {
            assert_eq!(spec.offset, offset, "{}", spec.name);
            assert_eq!(spec.shape.iter().product::<usize>(), spec.length, "{}", spec.name);
            offset += spec.length;
        }
        assert_eq!(offset, m::PARAMETER_COUNT);
        assert_eq!(m::WEIGHTS_INT6.len(), (m::PARAMETER_COUNT * 6).div_ceil(8));
    }

    #[test]
    fn codes_are_within_int6() {
        let levels = 31;
        for index in 0..m::PARAMETER_COUNT {
            let value = code(m::WEIGHTS_INT6, index);
            assert!((-levels..=levels).contains(&value), "{index}: {value}");
        }
    }
}
