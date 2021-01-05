use crate::{
    generate::ComputeShader,
    gpu_state::{DrawIndirect, GpuMeshLayer, GpuState},
    priority_cache::{Priority, PriorityCache, PriorityCacheEntry},
    terrain::{quadtree::VNode, tile_cache::TileCache},
};
use cgmath::Vector3;
use maplit::hashmap;
use serde::{Deserialize, Serialize};
use std::convert::TryInto;
use std::{
    mem,
    ops::{Index, IndexMut},
};
use vec_map::VecMap;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub(crate) enum MeshType {
    Grass = 0,
}
impl<T> Index<MeshType> for VecMap<T> {
    type Output = T;
    fn index(&self, i: MeshType) -> &Self::Output {
        &self[i as usize]
    }
}
impl<T> IndexMut<MeshType> for VecMap<T> {
    fn index_mut(&mut self, i: MeshType) -> &mut Self::Output {
        &mut self[i as usize]
    }
}

struct Entry {
    priority: Priority,
    node: VNode,
    valid: bool,
}
impl Entry {
    fn new(node: VNode, priority: Priority) -> Self {
        Self { node, priority, valid: false }
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
    _padding2: [u32; 55],
}
unsafe impl bytemuck::Zeroable for MeshNodeState {}
unsafe impl bytemuck::Pod for MeshNodeState {}

pub(crate) struct MeshCacheDesc {
    pub max_bytes_per_entry: u64,
    pub generate: ComputeShader<MeshGenerateUniforms>,
    pub render: rshader::ShaderSet,
    pub dimensions: u32,
    pub dependency_mask: u32,
    pub level: u8,
    pub ty: MeshType,
}

pub(crate) struct MeshCache {
    inner: PriorityCache<Entry>,
    desc: MeshCacheDesc,

    uniforms: wgpu::Buffer,
    bindgroup_pipeline: Option<(wgpu::BindGroup, wgpu::RenderPipeline)>,
}
impl MeshCache {
    pub fn new(device: &wgpu::Device, size: usize, desc: MeshCacheDesc) -> Self {
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            size: (mem::size_of::<MeshNodeState>() * size) as u64,
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
            label: Some("grass.uniforms"),
        });
        Self { inner: PriorityCache::new(size), desc, uniforms, bindgroup_pipeline: None }
    }

    pub fn make_buffers(&self, device: &wgpu::Device) -> GpuMeshLayer {
        let indirect = device.create_buffer(&wgpu::BufferDescriptor {
            size: (mem::size_of::<DrawIndirect>() * self.inner.size()) as u64,
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::INDIRECT | wgpu::BufferUsage::COPY_DST,
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

    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tile_cache: &TileCache,
        gpu_state: &GpuState,
        camera: mint::Point3<f64>,
    ) {
        let camera = Vector3::new(camera.x, camera.y, camera.z);

        // Update priorities
        for entry in self.inner.slots_mut() {
            entry.priority = entry.node.priority(camera);
        }
        let min_priority =
            self.inner.slots().iter().map(|s| s.priority).min().unwrap_or(Priority::none());

        // Find any tiles that may need to be added.
        let mut missing = Vec::new();
        VNode::breadth_first(|node| {
            let priority = node.priority(camera);
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

        if self.desc.generate.refresh() {
            for entry in self.inner.slots_mut() {
                entry.valid = false;
            }
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("grass.command_encoder"),
        });
        for (index, entry) in self.inner.slots_mut().into_iter().enumerate() {
            if entry.valid || !tile_cache.contains_all(entry.node, self.desc.dependency_mask) {
                continue;
            }

            let uniforms = MeshGenerateUniforms {
                input_slot: tile_cache.get_slot(entry.node).unwrap() as u32,
                output_slot: index as u32,
            };

            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                size: mem::size_of::<DrawIndirect>() as u64,
                usage: wgpu::BufferUsage::COPY_SRC,
                mapped_at_creation: true,
                label: Some("grass.clear_indirect.tmp"),
            });
            for b in &mut *buffer.slice(..).get_mapped_range_mut() {
                *b = 0;
            }
            buffer.unmap();
            encoder.copy_buffer_to_buffer(
                &buffer,
                0,
                &gpu_state.mesh_cache[self.desc.ty].indirect,
                (mem::size_of::<DrawIndirect>() * index) as u64,
                mem::size_of::<DrawIndirect>() as u64,
            );

            self.desc.generate.run(
                device,
                &mut encoder,
                gpu_state,
                (self.desc.dimensions, self.desc.dimensions, 1),
                &uniforms,
            );
            entry.valid = true;
        }
        queue.submit(Some(encoder.finish()));
    }

    pub fn render<'a>(
        &'a mut self,
        device: &wgpu::Device,
        queue: &'a wgpu::Queue,
        rpass: &mut wgpu::RenderPass<'a>,
        gpu_state: &'a GpuState,
        uniform_buffer: &wgpu::Buffer,
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
                    "ubo" => (false, wgpu::BindingResource::Buffer {
                        buffer: uniform_buffer,
                        offset: 0,
                        size: None,
                    }),
                    "node" => (true, wgpu::BindingResource::Buffer {
                        buffer: &self.uniforms,
                        offset: 0,
                        size: Some((mem::size_of::<MeshNodeState>() as u64).try_into().unwrap()),
                    })
                ],
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
                    vertex_stage: wgpu::ProgrammableStageDescriptor {
                        module: &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                            label: Some("grass.vertex_shader"),
                            source: wgpu::ShaderSource::SpirV(self.desc.render.vertex().into()),
                            flags: wgpu::ShaderFlags::empty(),
                        }),
                        entry_point: "main",
                    },
                    fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                        module: &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                            label: Some("grass.fragment_shader"),
                            source: wgpu::ShaderSource::SpirV(self.desc.render.fragment().into()),
                            flags: wgpu::ShaderFlags::empty(),
                        }),
                        entry_point: "main",
                    }),
                    rasterization_state: Some(Default::default()),
                    primitive_topology: wgpu::PrimitiveTopology::TriangleList,
                    color_states: &[wgpu::ColorStateDescriptor {
                        format: wgpu::TextureFormat::Bgra8UnormSrgb,
                        color_blend: wgpu::BlendDescriptor::REPLACE,
                        alpha_blend: wgpu::BlendDescriptor::REPLACE,
                        write_mask: wgpu::ColorWrite::ALL,
                    }],
                    depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::Greater,
                        stencil: Default::default(),
                    }),
                    vertex_state: wgpu::VertexStateDescriptor {
                        index_format: None,
                        vertex_buffers: &[],
                    },
                    sample_count: 1,
                    sample_mask: !0,
                    alpha_to_coverage_enabled: false,
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
                _padding1: 0.0,
                _padding2: [0; 55],
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