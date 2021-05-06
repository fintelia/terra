use maplit::hashmap;

use crate::GpuState;
use std::{collections::HashMap, mem};

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct GenHeightmapsUniforms {
    pub position: [i32; 2],
    pub origin: [i32; 2],
    pub spacing: f32,
    pub in_slot: i32,
    pub out_slot: i32,
    pub level_resolution: i32,
    pub face: u32,
}
unsafe impl bytemuck::Zeroable for GenHeightmapsUniforms {}
unsafe impl bytemuck::Pod for GenHeightmapsUniforms {}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct GenDisplacementsUniforms {
    pub node_center: [f64; 3],
    pub padding0: f64,
    pub origin: [i32; 2],
    pub position: [i32; 2],
    pub stride: i32,
    pub heightmaps_slot: i32,
    pub displacements_slot: i32,
    pub face: i32,
    pub level_resolution: u32,
}
unsafe impl bytemuck::Zeroable for GenDisplacementsUniforms {}
unsafe impl bytemuck::Pod for GenDisplacementsUniforms {}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct GenNormalsUniforms {
    pub heightmaps_origin: [i32; 2],
    pub heightmaps_slot: i32,
    pub normals_slot: i32,
    pub spacing: f32,
    pub padding: [f32; 3],
}
unsafe impl bytemuck::Zeroable for GenNormalsUniforms {}
unsafe impl bytemuck::Pod for GenNormalsUniforms {}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct GenMaterialsUniforms {
    pub heightmaps_origin: [i32; 2],
    pub parent_origin: [u32; 2],
    pub heightmaps_slot: i32,
    pub normals_slot: i32,
    pub albedo_slot: i32,
    pub parent_slot: i32,
    pub spacing: f32,
    pub padding: i32,
}
unsafe impl bytemuck::Zeroable for GenMaterialsUniforms {}
unsafe impl bytemuck::Pod for GenMaterialsUniforms {}

pub(crate) struct ComputeShader<U> {
    shader: rshader::ShaderSet,
    bindgroup_pipeline: Option<(wgpu::BindGroup, wgpu::ComputePipeline)>,
    uniforms: Option<wgpu::Buffer>,
    name: String,
    _phantom: std::marker::PhantomData<U>,
}
#[allow(unused)]
impl<U: bytemuck::Pod> ComputeShader<U> {
    pub fn new(shader: rshader::ShaderSource, name: String) -> Self {
        Self {
            shader: rshader::ShaderSet::compute_only(shader).unwrap(),
            bindgroup_pipeline: None,
            uniforms: None,
            name,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn refresh(&mut self) -> bool {
        if self.shader.refresh() {
            self.bindgroup_pipeline = None;
            true
        } else {
            false
        }
    }

    pub fn run(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        state: &GpuState,
        dimensions: (u32, u32, u32),
        uniforms: &U,
    ) {
        if self.uniforms.is_none() {
            self.uniforms = Some(device.create_buffer(&wgpu::BufferDescriptor {
                size: mem::size_of::<U>() as u64,
                usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::UNIFORM,
                mapped_at_creation: false,
                label: Some(&format!("buffer.{}.uniforms", self.name)),
            }));
        }
        if self.bindgroup_pipeline.is_none() {
            let (bind_group, bind_group_layout) = state.bind_group_for_shader(
                device,
                &self.shader,
                hashmap!["ubo".into() => (false, wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: self.uniforms.as_ref().unwrap(),
                    offset: 0,
                    size: None,
                }))],
                HashMap::new(),
                &format!("bindgroup.{}", self.name),
            );
            self.bindgroup_pipeline = Some((
                bind_group,
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        bind_group_layouts: [&bind_group_layout][..].into(),
                        push_constant_ranges: &[],
                        label: Some(&format!("pipeline.{}.layout", self.name)),
                    })),
                    module: &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                        label: Some(&format!("shader.{}", self.name)),
                        source: wgpu::ShaderSource::SpirV(self.shader.compute().into()),
                        flags: wgpu::ShaderFlags::empty(),
                    }),
                    entry_point: "main",
                    label: Some(&format!("pipeline.{}", self.name)),
                }),
            ));
        }

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            size: mem::size_of::<U>() as u64,
            usage: wgpu::BufferUsage::COPY_SRC,
            label: Some(&format!("buffer.temporary.{}.upload", self.name)),
            mapped_at_creation: true,
        });
        let mut buffer_view = staging.slice(..).get_mapped_range_mut();
        buffer_view[..mem::size_of::<U>()].copy_from_slice(bytemuck::bytes_of(uniforms));
        drop(buffer_view);
        staging.unmap();

        encoder.copy_buffer_to_buffer(
            &staging,
            0,
            self.uniforms.as_ref().unwrap(),
            0,
            mem::size_of::<U>() as u64,
        );

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&self.bindgroup_pipeline.as_ref().unwrap().1);
        cpass.set_bind_group(0, &self.bindgroup_pipeline.as_ref().unwrap().0, &[]);
        cpass.dispatch(dimensions.0, dimensions.1, dimensions.2);
    }
}
