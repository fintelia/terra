use crate::{
    cache::{tile::SLOTS_PER_LEVEL, LayerType, MeshType, Priority, PriorityCacheEntry},
    generate::ComputeShader,
    gpu_state::{DrawIndexedIndirect, GpuState},
};
use std::collections::HashMap;
use std::mem;
use wgpu::util::DeviceExt;

use super::{LayerMask, UnifiedPriorityCache};

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct MeshGenerateUniforms {
    slot: u32,
    storage_base_entry: u32,
    mesh_base_entry: u32,
    entries_per_node: u32,
}
unsafe impl bytemuck::Zeroable for MeshGenerateUniforms {}
unsafe impl bytemuck::Pod for MeshGenerateUniforms {}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct CullMeshUniforms {
    pub(super) base_entry: u32,
    pub(super) num_nodes: u32,
    pub(super) entries_per_node: u32,
    pub(super) base_slot: u32,
    pub(super) mesh_index: u32,
}
unsafe impl bytemuck::Zeroable for CullMeshUniforms {}
unsafe impl bytemuck::Pod for CullMeshUniforms {}

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct MeshNodeState {
    relative_position: [f32; 3],
    min_distance: f32,
    parent_relative_position: [f32; 3],
    _padding1: f32,

    slot: u32,
    face: u32,
    _padding2: [u32; 6],
}
unsafe impl bytemuck::Zeroable for MeshNodeState {}
unsafe impl bytemuck::Pod for MeshNodeState {}

pub(crate) struct MeshCacheDesc {
    pub max_bytes_per_node: u64,
    pub index_buffer: wgpu::Buffer,
    pub generate: Vec<((u32, u32, u32), ComputeShader<MeshGenerateUniforms>)>,
    pub render: rshader::ShaderSet,
    pub entries_per_node: usize,
    pub peer_dependency_mask: LayerMask,
    pub ancester_dependency_mask: LayerMask,
    pub min_level: u8,
    pub max_level: u8,
    pub ty: MeshType,
}

pub(crate) struct MeshCache {
    pub(super) desc: MeshCacheDesc,

    pub(super) base_entry: usize,
    pub(super) num_entries: usize,

    bindgroup_pipeline: Option<(wgpu::BindGroup, wgpu::RenderPipeline)>,
}
impl MeshCache {
    pub(super) fn new(desc: MeshCacheDesc, base_slot: usize, num_slots: usize) -> Self {
        Self { desc, base_entry: base_slot, num_entries: num_slots, bindgroup_pipeline: None }
    }

    pub(super) fn generate_all(
        cache: &mut UnifiedPriorityCache,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        gpu_state: &GpuState,
    ) {
        let tiles = &cache.tiles;
        let mut generated = Vec::new();
        let mut command_buffers = Vec::new();
        for mesh_type in MeshType::iter() {
            let m = &mut cache.meshes[mesh_type];
            let min_level = m.desc.min_level;

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("{}.command_encoder", mesh_type.name())),
            });

            let mut zero_buffer = None;
            for (index, entry) in (m.desc.min_level..=m.desc.max_level).flat_map(|l| {
                tiles.levels[l as usize]
                    .slots()
                    .into_iter()
                    .enumerate()
                    .map(move |(i, s)| ((l - min_level) as usize * SLOTS_PER_LEVEL + i, s))
            }) {
                if entry.valid.contains_mesh(mesh_type)
                    || entry.priority() < Priority::cutoff()
                    || entry.valid & m.desc.peer_dependency_mask != m.desc.peer_dependency_mask
                {
                    continue;
                }

                let ancester_dependency_mask = m.desc.ancester_dependency_mask;
                let has_all_ancestor_dependencies = LayerType::iter()
                    .filter(|layer| ancester_dependency_mask.contains_layer(*layer))
                    .all(|layer| {
                        if entry.node.level() < tiles.layers[layer].min_level {
                            false
                        } else if entry.node.level() <= tiles.layers[layer].max_level {
                            tiles.contains(entry.node, layer)
                        } else {
                            let ancestor = entry
                                .node
                                .find_ancestor(|node| node.level() == tiles.layers[layer].max_level)
                                .unwrap()
                                .0;
                            tiles.contains(ancestor, layer)
                        }
                    });
                if !has_all_ancestor_dependencies {
                    continue;
                }

                if zero_buffer.is_none() {
                    zero_buffer =
                        Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            usage: wgpu::BufferUsage::COPY_SRC,
                            label: Some(&format!("{}.clear_indirect.tmp", mesh_type.name())),
                            contents: &vec![0; mem::size_of::<DrawIndexedIndirect>() * 16],
                        }));
                }

                encoder.copy_buffer_to_buffer(
                    zero_buffer.as_ref().unwrap(),
                    0,
                    &gpu_state.mesh_indirect,
                    (m.base_entry
                        + mem::size_of::<DrawIndexedIndirect>() * index * m.desc.entries_per_node)
                        as u64,
                    (mem::size_of::<DrawIndexedIndirect>() * m.desc.entries_per_node) as u64,
                );

                let uniforms = MeshGenerateUniforms {
                    slot: cache.tiles.get_slot(entry.node).unwrap() as u32,
                    storage_base_entry: (index * m.desc.entries_per_node) as u32,
                    mesh_base_entry: (m.base_entry + index * m.desc.entries_per_node) as u32,
                    entries_per_node: m.desc.entries_per_node as u32,
                };

                for (dims, shader) in &mut m.desc.generate {
                    shader.run(device, &mut encoder, gpu_state, *dims, &uniforms);
                }

                generated.push((mesh_type, entry.node));
            }
            command_buffers.push(encoder.finish());
        }

        // TODO: Update generators
        for (mesh_type, node) in generated {
            cache.tiles.levels[node.level() as usize].entry_mut(&node).unwrap().valid |=
                mesh_type.bit_mask();
            // cache.meshes[mesh_type].inner.entry_mut(&node).unwrap().generators = cache
            //     .generator_dependencies(node, cache.meshes[mesh_type].desc.peer_dependency_mask);
        }

        queue.submit(command_buffers);
    }

    pub fn render<'a>(
        &'a mut self,
        device: &wgpu::Device,
        rpass: &mut wgpu::RenderPass<'a>,
        gpu_state: &'a GpuState,
    ) {
        if self.desc.render.refresh() {
            self.bindgroup_pipeline = None;
        }
        if self.bindgroup_pipeline.is_none() {
            let (bind_group, bind_group_layout) = gpu_state.bind_group_for_shader(
                device,
                &self.desc.render,
                HashMap::new(),
                HashMap::new(),
                "grass",
            );
            let render_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                    label: Some("grass.pipeline_layout"),
                });
            self.bindgroup_pipeline = Some((
                bind_group,
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    layout: Some(&render_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                            label: Some("grass.vertex_shader"),
                            source: wgpu::ShaderSource::SpirV(self.desc.render.vertex().into()),
                            flags: wgpu::ShaderFlags::empty(),
                        }),
                        entry_point: "main",
                        buffers: &[],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                            label: Some("grass.fragment_shader"),
                            source: wgpu::ShaderSource::SpirV(self.desc.render.fragment().into()),
                            flags: wgpu::ShaderFlags::empty(),
                        }),
                        entry_point: "main",
                        targets: &[wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Bgra8UnormSrgb,
                            blend: Some(wgpu::BlendState {
                                color: wgpu::BlendComponent::REPLACE,
                                alpha: wgpu::BlendComponent::REPLACE,
                            }),
                            write_mask: wgpu::ColorWrite::ALL,
                        }],
                    }),
                    primitive: Default::default(),
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::Greater,
                        bias: Default::default(),
                        stencil: Default::default(),
                    }),
                    multisample: Default::default(),
                    label: Some("grass.render_pipeline"),
                }),
            ));
        }

        rpass.set_pipeline(&self.bindgroup_pipeline.as_ref().unwrap().1);
        rpass.set_index_buffer(self.desc.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        rpass.set_bind_group(0, &self.bindgroup_pipeline.as_ref().unwrap().0, &[]);
        if device.features().contains(wgpu::Features::MULTI_DRAW_INDIRECT) {
            rpass.multi_draw_indexed_indirect(
                &gpu_state.mesh_indirect,
                (self.base_entry * mem::size_of::<DrawIndexedIndirect>()) as u64,
                self.num_entries as u32,
            );
        } else {
            for i in 0..self.num_entries {
                rpass.draw_indexed_indirect(
                    &gpu_state.mesh_indirect,
                    ((self.base_entry + i) * mem::size_of::<DrawIndexedIndirect>()) as u64,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_state_size() {
        assert_eq!(mem::size_of::<MeshNodeState>(), 256);
    }
}
