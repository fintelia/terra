use crate::{
    cache::{LayerType, MeshType, Priority, PriorityCache, PriorityCacheEntry, SLOTS_PER_LEVEL},
    generate::ComputeShader,
    gpu_state::{DrawIndexedIndirect, GpuMeshLayer, GpuState},
    terrain::quadtree::{QuadTree, VNode},
    utils::math::InfiniteFrustum,
};
use cgmath::Vector2;
use maplit::hashmap;
use std::collections::HashMap;
use std::mem;
use wgpu::util::DeviceExt;

use super::{GeneratorMask, LayerMask, TileCache, UnifiedPriorityCache};

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct MeshGenerateUniforms {
    texture_slot: u32,
    texture_step: f32,
    texture_origin: [f32; 2],

    tile_slot: u32,
    output_slot: u32,
    level: u32,
    padding: u32,
}
unsafe impl bytemuck::Zeroable for MeshGenerateUniforms {}
unsafe impl bytemuck::Pod for MeshGenerateUniforms {}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct CullMeshUniforms {
    num_nodes: u32,
    entries_per_node: u32,
    padding: [u32; 2],

    nodes: [([f32; 3], u32); 512],
}
unsafe impl bytemuck::Zeroable for CullMeshUniforms {}
unsafe impl bytemuck::Pod for CullMeshUniforms {}

impl Default for CullMeshUniforms {
    fn default() -> Self {
        Self { num_nodes: 0, entries_per_node: 16, padding: [0; 2], nodes: [([0.0; 3], 0); 512] }
    }
}

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
    pub max_bytes_per_entry: u64,
    pub index_buffer: wgpu::Buffer,
    pub generate: ComputeShader<MeshGenerateUniforms>,
    pub render: rshader::ShaderSet,
    pub dimensions: u32,
    pub peer_dependency_mask: LayerMask,
    pub ancester_dependency_mask: LayerMask,
    pub min_level: u8,
    pub max_level: u8,
    pub ty: MeshType,
}

pub(crate) struct MeshCache {
    pub(super) desc: MeshCacheDesc,

    nodes: wgpu::Buffer,
    bindgroup_pipeline: Option<(wgpu::BindGroup, wgpu::RenderPipeline)>,

    compute_bounds: ComputeShader<u32>,
}
impl MeshCache {
    pub(super) fn new(device: &wgpu::Device, desc: MeshCacheDesc) -> Self {
        assert!(desc.min_level >= 2);
        let num_slots =
            super::SLOTS_PER_LEVEL as u64 * (1 + desc.max_level - desc.min_level) as u64;

        let nodes = device.create_buffer(&wgpu::BufferDescriptor {
            size: mem::size_of::<MeshNodeState>() as u64 * num_slots,
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
            label: Some("grass.nodes_buffer"),
        });
        let compute_bounds = ComputeShader::new(
            rshader::shader_source!("../shaders", "bounding-sphere.comp", "declarations.glsl"),
            "bounding-sphere".to_owned(),
        );
        Self { desc, nodes, bindgroup_pipeline: None, compute_bounds }
    }

    pub(super) fn make_buffers(&self, device: &wgpu::Device) -> GpuMeshLayer {
        let num_slots =
            super::SLOTS_PER_LEVEL as u64 * (1 + self.desc.max_level - self.desc.min_level) as u64;

        let indirect = device.create_buffer(&wgpu::BufferDescriptor {
            size: mem::size_of::<DrawIndexedIndirect>() as u64 * num_slots * 16,
            usage: wgpu::BufferUsage::STORAGE
                | wgpu::BufferUsage::INDIRECT
                | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: true,
            label: Some("grass.indirect"),
        });
        for b in &mut *indirect.slice(..).get_mapped_range_mut() {
            *b = 0;
        }
        indirect.unmap();

        let bounding = device.create_buffer(&wgpu::BufferDescriptor {
            size: 16 * 16 * num_slots,
            usage: wgpu::BufferUsage::STORAGE
                | wgpu::BufferUsage::STORAGE
                | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: true,
            label: Some("grass.bounding"),
        });
        for b in &mut *bounding.slice(..).get_mapped_range_mut() {
            *b = 0;
        }
        bounding.unmap();

        GpuMeshLayer {
            indirect,
            bounding,
            storage: device.create_buffer(&wgpu::BufferDescriptor {
                size: self.desc.max_bytes_per_entry * num_slots,
                usage: wgpu::BufferUsage::STORAGE,
                mapped_at_creation: false,
                label: Some("grass.storage"),
            }),
        }
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
            'outer: for (index, entry) in (m.desc.min_level..=m.desc.max_level).flat_map(|l| {
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

                let mut texture_slot = None;
                if m.desc.ancester_dependency_mask.contains_layer(LayerType::GrassCanopy) {
                    assert_eq!(m.desc.ancester_dependency_mask, LayerType::GrassCanopy.bit_mask());

                    let texture_origin = Vector2::new(2.5, 2.5) / 516.0;
                    let texture_ratio = 511.0 / 516.0;
                    let texture_step = 511.0 / 516.0;
                    let (ancestor, generations, offset) = entry
                        .node
                        .find_ancestor(|n| {
                            n.level() <= tiles.layers[LayerType::GrassCanopy].max_level
                        })
                        .unwrap();
                    let scale = (0.5f32).powi(generations as i32);
                    let offset = Vector2::new(offset.x as f32, offset.y as f32);
                    let offset = texture_origin + scale * texture_ratio * offset;

                    match cache.tiles.get_slot(ancestor) {
                        None => continue 'outer,
                        Some(index) => {
                            assert!(texture_slot.is_none());
                            texture_slot =
                                Some((index, scale * texture_step, [offset.x, offset.y]));
                        }
                    }
                }
                let texture_slot = texture_slot.unwrap_or((0, 0.0, [0., 0.]));

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
                    &gpu_state.mesh_cache[m.desc.ty].indirect,
                    (mem::size_of::<DrawIndexedIndirect>() * index * 16) as u64,
                    mem::size_of::<DrawIndexedIndirect>() as u64 * 16,
                );

                m.desc.generate.run(
                    device,
                    &mut encoder,
                    gpu_state,
                    (m.desc.dimensions, m.desc.dimensions, 1),
                    &MeshGenerateUniforms {
                        texture_slot: texture_slot.0 as u32,
                        texture_origin: texture_slot.2,
                        texture_step: texture_slot.1,
                        tile_slot: cache.tiles.get_slot(entry.node).unwrap() as u32,
                        output_slot: index as u32 * 16,
                        level: entry.node.level() as u32,
                        padding: 0,
                    },
                );
                m.compute_bounds.run(
                    device,
                    &mut encoder,
                    gpu_state,
                    (16, 1, 1),
                    &(index as u32 * 16),
                );

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

    pub fn cull_meshes<'a>(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        gpu_state: &GpuState,
        tile_cache: &TileCache,
        camera: mint::Point3<f64>,
        frustum: &InfiniteFrustum,
        cull_shader: &mut ComputeShader<CullMeshUniforms>,
    ) {
        let num_slots =
            super::SLOTS_PER_LEVEL as u64 * (1 + self.desc.max_level - self.desc.min_level) as u64;

        let mut cull_ubo = CullMeshUniforms::default();
        cull_ubo.num_nodes = num_slots as u32;
        for level in self.desc.min_level..=self.desc.max_level {
            for (i, entry) in tile_cache.levels[level as usize].slots().into_iter().enumerate() {
                cull_ubo.nodes[(level - self.desc.min_level) as usize * SLOTS_PER_LEVEL + i] = (
                    (cgmath::Point3::from(camera) - entry.node.center_wspace())
                        .cast::<f32>()
                        .unwrap()
                        .into(),
                    (entry.valid.contains_mesh(self.desc.ty)/*&& entry.priority() > Priority::cutoff()
                    && entry.node.in_frustum(frustum, tile_cache.get_height_range(entry.node))*/)
                        as u32,
                );
            }
        }
        cull_shader.run(
            device,
            encoder,
            &gpu_state,
            ((num_slots as u32 * 16 + 63) / 64, 1, 1),
            &cull_ubo,
        );
    }

    pub fn render<'a>(
        &'a mut self,
        tile_cache: &TileCache,
        device: &wgpu::Device,
        queue: &'a wgpu::Queue,
        rpass: &mut wgpu::RenderPass<'a>,
        gpu_state: &'a GpuState,
        camera: mint::Point3<f64>,
    ) {
        if self.desc.render.refresh() {
            self.bindgroup_pipeline = None;
        }
        if self.bindgroup_pipeline.is_none() {
            let (bind_group, bind_group_layout) = gpu_state.bind_group_for_shader(
                device,
                &self.desc.render,
                hashmap![
                    "nodes".into() => (false, wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.nodes,
                        offset: 0,
                        size: None,
                    }))
                ],
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

        // // Compute attributes for each node.
        let mut nodes = vec![Default::default(); (1 + self.desc.max_level - self.desc.min_level) as usize * SLOTS_PER_LEVEL];
        for level in self.desc.min_level..=self.desc.max_level {
            for (i, entry) in tile_cache.levels[level as usize].slots().into_iter().enumerate() {
                let slot = (level - self.desc.min_level) as usize * SLOTS_PER_LEVEL + i;
                nodes[slot] = MeshNodeState {
                    relative_position: (cgmath::Point3::from(camera) - entry.node.center_wspace())
                        .cast::<f32>()
                        .unwrap()
                        .into(),
                    parent_relative_position: (cgmath::Point3::from(camera)
                        - entry.node.parent().map(|x| x.0).unwrap_or(entry.node).center_wspace())
                    .cast::<f32>()
                    .unwrap()
                    .into(),
                    min_distance: entry.node.min_distance() as f32,
                    slot: slot as u32,
                    face: entry.node.face() as u32,
                    _padding1: 0.0,
                    _padding2: [0; 6],
                };
            }
        }

        if !nodes.is_empty() {
            queue.write_buffer(&self.nodes, 0, bytemuck::cast_slice(&nodes));
            rpass.set_pipeline(&self.bindgroup_pipeline.as_ref().unwrap().1);
            rpass.set_index_buffer(self.desc.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            rpass.set_bind_group(0, &self.bindgroup_pipeline.as_ref().unwrap().0, &[]);
            if device.features().contains(wgpu::Features::MULTI_DRAW_INDIRECT) {
                rpass.multi_draw_indexed_indirect(
                    &gpu_state.mesh_cache[self.desc.ty].indirect,
                    0,
                    nodes.len() as u32 * 16,
                );
            } else {
                for i in 0..(nodes.len() * 16) {
                    rpass.draw_indexed_indirect(
                        &gpu_state.mesh_cache[self.desc.ty].indirect,
                        i as u64 * mem::size_of::<DrawIndexedIndirect>() as u64,
                    );
                }
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
