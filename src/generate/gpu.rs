use maplit::hashmap;

use crate::GpuState;
use std::{collections::HashMap, mem};

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

    pub fn refresh(&mut self, device: &wgpu::Device, gpu_state: &GpuState) -> bool {
        if mem::size_of::<U>() > 0 && self.uniforms.is_none() {
            self.uniforms = Some(device.create_buffer(&wgpu::BufferDescriptor {
                size: mem::size_of::<U>() as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
                mapped_at_creation: false,
                label: Some(&format!("buffer.{}.uniforms", &self.name)),
            }));
        }

        let refreshed = self.shader.refresh();

        if refreshed || self.bindgroup_pipeline.is_none() {
            let (bind_group, bind_group_layout) = gpu_state.bind_group_for_shader(
                device,
                &self.shader,
                if self.uniforms.is_some() {
                    hashmap!["ubo".into() => (false, wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: self.uniforms.as_ref().unwrap(),
                        offset: 0,
                        size: None,
                    }))]
                } else {
                    HashMap::new()
                },
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
                        source: self.shader.compute(),
                    }),
                    entry_point: "main",
                    label: Some(&format!("pipeline.{}", self.name)),
                }),
            ))
        }

        refreshed
    }

    pub fn run(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        state: &GpuState,
        dimensions: (u32, u32, u32),
        uniforms: &U,
    ) {
        if self.uniforms.is_some() {
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                size: mem::size_of::<U>() as u64,
                usage: wgpu::BufferUsages::COPY_SRC,
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
        }

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&self.bindgroup_pipeline.as_ref().unwrap().1);
        cpass.set_bind_group(0, &self.bindgroup_pipeline.as_ref().unwrap().0, &[]);
        cpass.dispatch(dimensions.0, dimensions.1, dimensions.2);
    }
}
