//! The GPU backend (feature `gpu`): upstream's promoted WGSL compute
//! shader on wgpu, driven as upstream `gpu.js` drives it on WebGPU.
//!
//! The shader, its entry points and the runtime weight buffer come from
//! upstream (`scripts/sync-upstream.mjs`), with one fix to the shader: the
//! in-block top-down pass raced on workgroup memory (see the script). Storage is `f16` when
//! the adapter has `SHADER_F16`, else the same shader with `f32` storage.
//! The device, pipelines and weights are made once per process, on the
//! first call that wants the GPU; without an adapter every call falls back
//! to the CPU.

use std::sync::{Mutex, OnceLock};

use wgpu::util::DeviceExt;

use crate::generated::runtime;
use crate::prepare::Prepared;

const BLOCK: u32 = 32;
const HIDDEN: u64 = runtime::HIDDEN as u64;
/// Batches over this many blocks classify in up to eight chunks.
const CHUNK_MIN_BLOCKS: u32 = 4096;

/// Expands the runtime weight symbols into the floats the shader reads
/// (upstream `weights.js`).
pub fn runtime_weights() -> Vec<f32> {
    let symbols = runtime::WEIGHTS.as_bytes();
    let mut out = Vec::with_capacity(symbols.len());
    let mut start = 0;
    for &(length, scale) in &runtime::WEIGHT_RANGES {
        for &code in &symbols[start..start + length] {
            let code = i32::from(code);
            let value = if code >= 97 {
                code - 71
            } else if code >= 65 {
                code - 65
            } else if code >= 48 {
                code + 4
            } else if code == 45 {
                62
            } else {
                63
            };
            let signed = if value & 1 != 0 {
                -f64::from(value + 1) / 2.0
            } else {
                f64::from(value) / 2.0
            };
            out.push((signed * scale) as f32);
        }
        start += length;
    }
    out
}

struct Runtime {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Vec<wgpu::ComputePipeline>,
    weights: wgpu::Buffer,
    f16: bool,
    adapter: String,
}

fn runtime() -> Option<&'static Mutex<Runtime>> {
    static RUNTIME: OnceLock<Option<Mutex<Runtime>>> = OnceLock::new();
    RUNTIME
        .get_or_init(|| {
            if std::env::var_os("NSH_GPU_LEXER_NO_GPU").is_some() {
                return None;
            }
            pollster::block_on(Runtime::new()).map(Mutex::new)
        })
        .as_ref()
}

/// Whether a GPU adapter was found (initializing the device on first use).
pub fn available() -> bool {
    runtime().is_some()
}

/// The adapter's name and whether it runs the f16 shader.
pub fn adapter_info() -> Option<(String, bool)> {
    runtime().map(|r| {
        let r = r.lock().unwrap();
        (r.adapter.clone(), r.f16)
    })
}

impl Runtime {
    async fn new() -> Option<Runtime> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                ..Default::default()
            })
            .await
            .ok()?;
        let f16 = adapter.features().contains(wgpu::Features::SHADER_F16)
            && std::env::var_os("NSH_GPU_LEXER_NO_F16").is_none();
        let adapter_limits = adapter.limits();
        let limits = wgpu::Limits {
            max_buffer_size: adapter_limits.max_buffer_size,
            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
            ..wgpu::Limits::default()
        };
        if adapter_limits.max_compute_workgroup_storage_size < 16_384
            || adapter_limits.max_storage_buffers_per_shader_stage < 8
        {
            return None;
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("nsh-gpu-lexer"),
                required_features: if f16 {
                    wgpu::Features::SHADER_F16
                } else {
                    wgpu::Features::empty()
                },
                required_limits: limits,
                ..Default::default()
            })
            .await
            .ok()?;
        let source = if f16 { runtime::SHADER_F16 } else { runtime::SHADER_F32 };
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu-lexer tree"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipelines = runtime::ENTRY_POINTS
            .chars()
            .map(|entry| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("gpu-lexer pass"),
                    layout: None,
                    module: &module,
                    entry_point: Some(&entry.to_string()),
                    compilation_options: Default::default(),
                    cache: None,
                })
            })
            .collect();
        let weights = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu-lexer weights"),
            contents: bytemuck::cast_slice(&runtime_weights()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        Some(Runtime {
            device,
            queue,
            pipelines,
            weights,
            f16,
            adapter: adapter.get_info().name,
        })
    }

    fn labels(&self, prepared: &Prepared) -> Option<Vec<u8>> {
        let device = &self.device;
        let tokens = prepared.len() as u32;
        let blocks = tokens.div_ceil(BLOCK);
        let power = blocks.next_power_of_two();
        let tree_nodes = power * 2 - 1;
        // One stream: [part_start, part_count, block_start, block_count, tree_offset, tree_power].
        let streams = [0, tokens, 0, blocks, 0, power];
        let mut block_table = Vec::with_capacity(blocks as usize * 4);
        for block in 0..blocks {
            let start = block * BLOCK;
            block_table.extend_from_slice(&[start, BLOCK.min(tokens - start), 0, block]);
        }
        let value_bytes = if self.f16 { 2 } else { 4 };
        let label_bytes = u64::from((tokens + 3) & !3);
        let limit = device.limits().max_storage_buffer_binding_size;
        let leaf_bytes = u64::from(tokens) * HIDDEN * value_bytes;
        let tree_bytes = u64::from(tree_nodes) * HIDDEN * value_bytes;
        let scratch_bytes = (u64::from(blocks) * HIDDEN * 16).max(u64::from(tree_nodes) * HIDDEN * 4);
        if [leaf_bytes, tree_bytes, scratch_bytes].iter().any(|&b| b > limit) {
            return None;
        }

        use wgpu::BufferUsages as U;
        let init = |label: &str, contents: &[u8], usage: U| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage,
            })
        };
        let empty = |label: &str, size: u64, usage: U| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(16).next_multiple_of(4),
                usage,
                mapped_at_creation: false,
            })
        };
        let features = init("features", bytemuck::cast_slice(&prepared.data), U::STORAGE);
        let labels = empty("labels", label_bytes, U::STORAGE | U::COPY_SRC | U::COPY_DST);
        let params = empty("params", 16, U::UNIFORM | U::COPY_DST);
        let stream_buffer = init("streams", bytemuck::cast_slice(&streams), U::STORAGE);
        let block_buffer = init("blocks", bytemuck::cast_slice(&block_table), U::STORAGE);
        let leaf_states = empty("leaf states", leaf_bytes, U::STORAGE);
        let tree_up = empty("tree up", tree_bytes, U::STORAGE);
        let scratch = empty("scratch", scratch_bytes, U::STORAGE);
        let neighbors = empty("neighbors", u64::from(blocks) * 8, U::STORAGE);
        let readback = empty("readback", label_bytes, U::MAP_READ | U::COPY_DST);

        // The buffers each pass binds, as upstream `createTreeBindGroups`.
        let bindings: [&[(u32, &wgpu::Buffer)]; 7] = [
            &[(0, &features), (2, &params), (9, &block_buffer), (12, &neighbors)],
            &[(2, &params), (3, &stream_buffer), (12, &neighbors)],
            &[
                (0, &features),
                (2, &params),
                (3, &stream_buffer),
                (4, &self.weights),
                (9, &block_buffer),
                (11, &scratch),
                (12, &neighbors),
            ],
            &[(2, &params), (3, &stream_buffer), (11, &scratch)],
            &[
                (0, &features),
                (2, &params),
                (3, &stream_buffer),
                (4, &self.weights),
                (5, &leaf_states),
                (6, &tree_up),
                (9, &block_buffer),
                (11, &scratch),
                (12, &neighbors),
            ],
            &[
                (2, &params),
                (3, &stream_buffer),
                (4, &self.weights),
                (6, &tree_up),
                (11, &scratch),
            ],
            &[
                (0, &features),
                (1, &labels),
                (2, &params),
                (4, &self.weights),
                (13, &self.weights),
                (3, &stream_buffer),
                (5, &leaf_states),
                (9, &block_buffer),
                (11, &scratch),
            ],
        ];
        let groups: Vec<wgpu::BindGroup> = bindings
            .iter()
            .enumerate()
            .map(|(index, entries)| {
                let entries: Vec<wgpu::BindGroupEntry> = entries
                    .iter()
                    .map(|&(binding, buffer)| wgpu::BindGroupEntry {
                        binding,
                        resource: buffer.as_entire_binding(),
                    })
                    .collect();
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.pipelines[index].get_bind_group_layout(0),
                    entries: &entries,
                })
            })
            .collect();

        let stream_count = 1u32;
        let chunk_blocks = CHUNK_MIN_BLOCKS.max(blocks.div_ceil(8));
        let mut first = 0;
        while first < blocks {
            let end = blocks.min(first + chunk_blocks);
            // The first submission runs every earlier pass, over all blocks.
            let block_count = if first == 0 { blocks } else { end };
            self.queue.write_buffer(
                &params,
                0,
                bytemuck::cast_slice(&[stream_count, first, block_count, tree_nodes]),
            );
            let mut encoder = device.create_command_encoder(&Default::default());
            if first == 0 {
                encoder.clear_buffer(&labels, 0, None);
            }
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                let mut encode = |index: usize, workgroups: u32| {
                    pass.set_pipeline(&self.pipelines[index]);
                    pass.set_bind_group(0, &groups[index], &[]);
                    let x = workgroups.clamp(1, 65_535);
                    pass.dispatch_workgroups(x, workgroups.div_ceil(x), 1);
                };
                if first == 0 {
                    let counts = [
                        (blocks + 63) >> 6,
                        stream_count,
                        blocks,
                        stream_count * 4,
                        blocks,
                        stream_count,
                    ];
                    for (index, count) in counts.into_iter().enumerate() {
                        encode(index, count);
                    }
                }
                encode(6, end - first);
            }
            if end == blocks {
                encoder.copy_buffer_to_buffer(&labels, 0, &readback, 0, label_bytes);
            }
            self.queue.submit([encoder.finish()]);
            first = end;
        }

        let slice = readback.slice(..label_bytes);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
        receiver.recv().ok()?.ok()?;
        let out = slice.get_mapped_range().ok()?[..tokens as usize].to_vec();
        readback.unmap();
        Some(out)
    }
}

/// Labels from the GPU, or `None` without one (or for an input over the
/// device's buffer limits), when the caller uses the CPU.
pub fn labels(prepared: &Prepared) -> Option<Vec<u8>> {
    let runtime = runtime()?;
    runtime.lock().ok()?.labels(prepared)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shaders_carry_the_down_pass_fix() {
        // scripts/sync-upstream.mjs patches the in-block top-down race.
        assert!(runtime::SHADER_F16.contains("nshL") && runtime::SHADER_F32.contains("nshL"));
    }

    #[test]
    fn runtime_weights_decode() {
        let weights = runtime_weights();
        assert_eq!(weights.len(), runtime::WEIGHTS.len());
        assert!(weights.iter().all(|w| w.is_finite()));
    }
}
