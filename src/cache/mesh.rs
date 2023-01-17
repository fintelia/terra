use crate::gpu_state::{DrawIndexedIndirect, GpuState};
use std::mem;
use std::{collections::HashMap, ops::Range};
use crate::cache::layer::MeshType;

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct MeshGenerateUniforms {
    pub(super) slot: u32,
    pub(super) storage_base_entry: u32,
    pub(super) mesh_base_entry: u32,
    pub(super) entries_per_node: u32,
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
    pub index_buffer: Vec<u32>,
    pub render: rshader::ShaderSet,
    pub render_shadow: Option<rshader::ShaderSet>,
    pub cull_mode: Option<wgpu::Face>,
    pub render_overlapping_levels: bool,
    pub entries_per_node: usize,
    pub min_level: u8,
    pub max_level: u8,
    pub ty: MeshType,
}

pub(crate) struct MeshCache {
    pub(super) desc: MeshCacheDesc,

    pub(super) base_entry: usize,
    pub(super) num_entries: usize,

    index_buffer_range: Range<u64>,

    bindgroup_pipeline: Option<(wgpu::BindGroup, wgpu::RenderPipeline)>,
    shadow_bindgroup_pipeline: Option<(wgpu::BindGroup, wgpu::RenderPipeline)>,
}
impl MeshCache {
    pub(super) fn new(
        desc: MeshCacheDesc,
        base_slot: usize,
        num_slots: usize,
        index_buffer_range: Range<u64>,
    ) -> Self {
        Self {
            desc,
            base_entry: base_slot,
            num_entries: num_slots,
            bindgroup_pipeline: None,
            shadow_bindgroup_pipeline: None,
            index_buffer_range,
        }
    }

    pub fn update(&mut self, device: &wgpu::Device, gpu_state: &GpuState) {
        if self.desc.render.refresh() {
            self.bindgroup_pipeline = None;
        }
        if self.bindgroup_pipeline.is_none() {
            let (bind_group, bind_group_layout) = gpu_state.bind_group_for_shader(
                device,
                &self.desc.render,
                HashMap::new(),
                HashMap::new(),
                self.desc.ty.name(),
            );
            let render_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                    label: Some(&format!("{}.pipeline_layout", self.desc.ty.name())),
                });
            self.bindgroup_pipeline = Some((
                bind_group,
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    layout: Some(&render_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                            label: Some(&format!("shader.{}.vertex", self.desc.ty.name())),
                            source: self.desc.render.vertex(),
                        }),
                        entry_point: "main",
                        buffers: &[],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                            label: Some(&format!("shader.{}.fragment", self.desc.ty.name())),
                            source: self.desc.render.fragment(),
                        }),
                        entry_point: "main",
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Bgra8UnormSrgb,
                            blend: Some(wgpu::BlendState {
                                color: wgpu::BlendComponent::REPLACE,
                                alpha: wgpu::BlendComponent::REPLACE,
                            }),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: wgpu::PrimitiveState {
                        cull_mode: self.desc.cull_mode,
                        ..Default::default()
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::Greater,
                        bias: Default::default(),
                        stencil: Default::default(),
                    }),
                    multisample: Default::default(),
                    multiview: None,
                    label: Some(&format!("pipeline.render.{}", self.desc.ty.name())),
                }),
            ));
        }

        if let Some(ref mut render_shadow) = self.desc.render_shadow {
            if render_shadow.refresh() {
                self.shadow_bindgroup_pipeline = None;
            }
            if self.shadow_bindgroup_pipeline.is_none() {
                let (bind_group, bind_group_layout) = gpu_state.bind_group_for_shader(
                    device,
                    &render_shadow,
                    HashMap::new(),
                    HashMap::new(),
                    self.desc.ty.name(),
                );
                let render_pipeline_layout =
                    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        bind_group_layouts: &[&bind_group_layout],
                        push_constant_ranges: &[],
                        label: Some(&format!("{}_shadow.pipeline_layout", self.desc.ty.name())),
                    });
                self.shadow_bindgroup_pipeline = Some((
                    bind_group,
                    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        layout: Some(&render_pipeline_layout),
                        vertex: wgpu::VertexState {
                            module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                                label: Some(&format!(
                                    "shader.{}_shadow.vertex",
                                    self.desc.ty.name()
                                )),
                                source: render_shadow.vertex(),
                            }),
                            entry_point: "main",
                            buffers: &[],
                        },
                        fragment: Some(wgpu::FragmentState {
                            module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                                label: Some(&format!(
                                    "shader.{}_shadow.fragment",
                                    self.desc.ty.name()
                                )),
                                source: render_shadow.fragment(),
                            }),
                            entry_point: "main",
                            targets: &[],
                        }),
                        primitive: wgpu::PrimitiveState {
                            cull_mode: self.desc.cull_mode,
                            ..Default::default()
                        },
                        depth_stencil: Some(wgpu::DepthStencilState {
                            format: wgpu::TextureFormat::Depth24Plus,
                            depth_write_enabled: true,
                            depth_compare: wgpu::CompareFunction::Less,
                            bias: wgpu::DepthBiasState {
                                constant: 0,
                                slope_scale: 0.0,
                                clamp: 0.0,
                            },
                            stencil: Default::default(),
                        }),
                        multisample: Default::default(),
                        multiview: None,
                        label: Some(&format!("pipeline.render.{}_shadow", self.desc.ty.name())),
                    }),
                ));
            }
        }
    }

    pub fn render<'a>(
        &'a self,
        device: &wgpu::Device,
        rpass: &mut wgpu::RenderPass<'a>,
        gpu_state: &'a GpuState,
    ) {
        rpass.set_pipeline(&self.bindgroup_pipeline.as_ref().unwrap().1);
        rpass.set_index_buffer(
            gpu_state.mesh_index.slice(self.index_buffer_range.clone()),
            wgpu::IndexFormat::Uint32,
        );
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

    pub fn render_shadow<'a>(
        &'a self,
        device: &wgpu::Device,
        rpass: &mut wgpu::RenderPass<'a>,
        gpu_state: &'a GpuState,
    ) {
        if self.desc.render_shadow.is_some() {
            rpass.set_pipeline(&self.shadow_bindgroup_pipeline.as_ref().unwrap().1);
            rpass.set_index_buffer(
                gpu_state.mesh_index.slice(self.index_buffer_range.clone()),
                wgpu::IndexFormat::Uint32,
            );
            rpass.set_bind_group(0, &self.shadow_bindgroup_pipeline.as_ref().unwrap().0, &[]);
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_state_size() {
        assert_eq!(mem::size_of::<MeshNodeState>(), 64);
    }
}
