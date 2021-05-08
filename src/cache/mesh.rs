use crate::{
    cache::{MeshType, Priority, PriorityCache, PriorityCacheEntry},
    generate::ComputeShader,
    gpu_state::{DrawIndirect, GpuMeshLayer, GpuState},
    terrain::quadtree::{QuadTree, VNode},
};
use maplit::hashmap;
use std::mem;
use std::{collections::HashMap, convert::TryInto};
use wgpu::util::DeviceExt;

use super::{GeneratorMask, LayerMask, UnifiedPriorityCache};

pub(super) struct Entry {
    priority: Priority,
    node: VNode,
    pub(super) valid: bool,
    pub(super) generators: GeneratorMask,
}
impl Entry {
    fn new(node: VNode, priority: Priority) -> Self {
        Self { node, priority, valid: false, generators: GeneratorMask::empty() }
    }
}
impl PriorityCacheEntry for Entry {
    type Key = VNode;
    fn priority(&self) -> Priority {
        self.priority
    }
    fn key(&self) -> VNode {
        self.node
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct MeshGenerateUniforms {
    input_slot: u32,
    output_slot: u32,
}
unsafe impl bytemuck::Zeroable for MeshGenerateUniforms {}
unsafe impl bytemuck::Pod for MeshGenerateUniforms {}

#[repr(C)]
#[derive(Copy, Clone)]
struct MeshNodeState {
    relative_position: [f32; 3],
    min_distance: f32,
    parent_relative_position: [f32; 3],
    _padding1: f32,

    slot: u32,
    face: u32,
    _padding2: [u32; 54],
}
unsafe impl bytemuck::Zeroable for MeshNodeState {}
unsafe impl bytemuck::Pod for MeshNodeState {}

pub(crate) struct MeshCacheDesc {
    pub max_bytes_per_entry: u64,
    pub generate: ComputeShader<MeshGenerateUniforms>,
    pub render: rshader::ShaderSet,
    pub dimensions: u32,
    pub dependency_mask: LayerMask,
    pub level: u8,
    pub ty: MeshType,
    pub size: usize,
}

pub(crate) struct MeshCache {
    pub(super) inner: PriorityCache<Entry>,
    pub(super) desc: MeshCacheDesc,

    uniforms: wgpu::Buffer,
    bindgroup_pipeline: Option<(wgpu::BindGroup, wgpu::RenderPipeline)>,
}
impl MeshCache {
    pub(super) fn new(device: &wgpu::Device, desc: MeshCacheDesc) -> Self {
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            size: (mem::size_of::<MeshNodeState>() * desc.size) as u64,
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
            label: Some("grass.uniforms"),
        });
        Self { inner: PriorityCache::new(desc.size), desc, uniforms, bindgroup_pipeline: None }
    }

    pub(super) fn make_buffers(&self, device: &wgpu::Device) -> GpuMeshLayer {
        let indirect = device.create_buffer(&wgpu::BufferDescriptor {
            size: (mem::size_of::<DrawIndirect>() * self.inner.size()) as u64,
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

        GpuMeshLayer {
            indirect,
            storage: device.create_buffer(&wgpu::BufferDescriptor {
                size: self.desc.max_bytes_per_entry * self.inner.size() as u64,
                usage: wgpu::BufferUsage::STORAGE,
                mapped_at_creation: false,
                label: Some("grass.storage"),
            }),
        }
    }

    pub(super) fn update(&mut self, quadtree: &QuadTree) {
        // Update priorities
        for entry in self.inner.slots_mut() {
            entry.priority = quadtree.node_priority(entry.node);
        }
        let min_priority =
            self.inner.slots().iter().map(|s| s.priority).min().unwrap_or(Priority::none());

        // Find any tiles that may need to be added.
        let mut missing = Vec::new();
        VNode::breadth_first(|node| {
            let priority = quadtree.node_priority(node);
            if priority < Priority::cutoff() {
                return false;
            }
            if node.level() == self.desc.level
                && !self.inner.contains(&node)
                && (priority > min_priority || !self.inner.is_full())
            {
                missing.push(Entry::new(node, priority));
            }

            node.level() < self.desc.level
        });
        self.inner.insert(missing);
    }

    pub(super) fn generate_all(
        cache: &mut UnifiedPriorityCache,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        gpu_state: &GpuState,
    ) {
        let mut generated = Vec::new();
        let mut command_buffers = Vec::new();
        for mesh_type in MeshType::iter() {
            let m = &mut cache.meshes[mesh_type];

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("{}.command_encoder", mesh_type.name())),
            });

            let mut zero_buffer = None;
            for (index, entry) in m.inner.slots_mut().into_iter().enumerate() {
                if entry.valid || entry.priority < Priority::cutoff() {
                    continue;
                }
                if !cache.tiles.contains_all(entry.node, m.desc.dependency_mask) {
                    continue;
                }

                if zero_buffer.is_none() {
                    zero_buffer =
                        Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            usage: wgpu::BufferUsage::COPY_SRC,
                            label: Some(&format!("{}.clear_indirect.tmp", mesh_type.name())),
                            contents: &vec![0; mem::size_of::<DrawIndirect>()],
                        }));
                }

                encoder.copy_buffer_to_buffer(
                    zero_buffer.as_ref().unwrap(),
                    0,
                    &gpu_state.mesh_cache[m.desc.ty].indirect,
                    (mem::size_of::<DrawIndirect>() * index) as u64,
                    mem::size_of::<DrawIndirect>() as u64,
                );

                m.desc.generate.run(
                    device,
                    &mut encoder,
                    gpu_state,
                    (m.desc.dimensions, m.desc.dimensions, 1),
                    &MeshGenerateUniforms {
                        input_slot: cache.tiles.get_slot(entry.node).unwrap() as u32,
                        output_slot: index as u32,
                    },
                );
                entry.valid = true;
                generated.push((mesh_type, entry.node));
            }
            command_buffers.push(encoder.finish());
        }
        for (mesh_type, node) in generated {
            cache.meshes[mesh_type].inner.entry_mut(&node).unwrap().generators =
                cache.generator_dependencies(node, cache.meshes[mesh_type].desc.dependency_mask);
        }

        queue.submit(command_buffers);
    }

    pub fn render<'a>(
        &'a mut self,
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
                    "node".into() => (true, wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.uniforms,
                        offset: 0,
                        size: Some((mem::size_of::<MeshNodeState>() as u64).try_into().unwrap()),
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

        // Compute attributes for each node.
        let nodes: Vec<_> = self
            .inner
            .slots()
            .into_iter()
            .enumerate()
            .filter(|e| e.1.valid && e.1.priority > Priority::cutoff())
            // .filter_map(|(i, e)| tile_cache.get_slot(e.node).map(|s| (i, s, e)))
            // .map(|(i, j, &Entry { node, .. })| MeshNodeState {
            .map(|(i, &Entry { node, .. })| MeshNodeState {
                relative_position: (cgmath::Point3::from(camera) - node.center_wspace())
                    .cast::<f32>()
                    .unwrap()
                    .into(),
                parent_relative_position: (cgmath::Point3::from(camera)
                    - node.parent().map(|x| x.0).unwrap_or(node).center_wspace())
                .cast::<f32>()
                .unwrap()
                .into(),
                min_distance: node.min_distance() as f32,
                slot: i as u32,
                face: node.face() as u32,
                //tile_slot: j as u32,
                _padding1: 0.0,
                _padding2: [0; 54],
            })
            .collect();

        if !nodes.is_empty() {
            queue.write_buffer(&self.uniforms, 0, bytemuck::cast_slice(&nodes));
            rpass.set_pipeline(&self.bindgroup_pipeline.as_ref().unwrap().1);
            for (i, node_state) in nodes.into_iter().enumerate() {
                rpass.set_bind_group(
                    0,
                    &self.bindgroup_pipeline.as_ref().unwrap().0,
                    &[(i * mem::size_of::<MeshNodeState>()) as u32],
                );
                rpass.draw_indirect(
                    &gpu_state.mesh_cache[self.desc.ty].indirect,
                    node_state.slot as u64 * mem::size_of::<DrawIndirect>() as u64,
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
