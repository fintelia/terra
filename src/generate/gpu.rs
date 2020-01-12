use crate::GpuState;
use std::mem;

#[derive(Copy, Clone)]
pub(crate) struct GenHeightsUniforms {
    pub position: [f32; 2],
    pub base_heights_step: f32,
    pub step: f32,
    pub slot: i32,
}
unsafe impl bytemuck::Zeroable for GenHeightsUniforms {}
unsafe impl bytemuck::Pod for GenHeightsUniforms {}

#[derive(Copy, Clone)]
pub(crate) struct GenNormalsUniforms {
    pub position: [f32; 2],
    pub spacing: f32,
    pub slot: i32,
}
unsafe impl bytemuck::Zeroable for GenNormalsUniforms {}
unsafe impl bytemuck::Pod for GenNormalsUniforms {}

pub(crate) struct ComputeShader<U> {
    shader: rshader::ShaderSet,
    bindgroup_pipeline: Option<(wgpu::BindGroup, wgpu::ComputePipeline)>,
    uniforms: wgpu::Buffer,
    _phantom: std::marker::PhantomData<U>,
}
impl<U: bytemuck::Pod> ComputeShader<U> {
    pub fn new(device: &wgpu::Device, shader: rshader::ShaderSet) -> Self {
        Self {
            shader,
            bindgroup_pipeline: None,
            uniforms: device.create_buffer(&wgpu::BufferDescriptor {
                size: mem::size_of::<U>() as u64,
                usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::UNIFORM,
            }),
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn refresh(&mut self, watcher: &mut rshader::ShaderDirectoryWatcher) -> bool {
        if self.shader.refresh(watcher) {
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
        if self.bindgroup_pipeline.is_none() {
            let (bind_group, bind_group_layout) = state.bind_group_for_shader(
                device,
                &self.shader,
                Some(&wgpu::BindingResource::Buffer {
                    buffer: &self.uniforms,
                    range: 0..mem::size_of::<U>() as u64,
                }),
            );
            self.bindgroup_pipeline = Some((
                bind_group,
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    layout: &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        bind_group_layouts: &[&bind_group_layout],
                    }),
                    compute_stage: wgpu::ProgrammableStageDescriptor {
                        module: &device.create_shader_module(
                            &wgpu::read_spirv(std::io::Cursor::new(self.shader.compute())).unwrap(),
                        ),
                        entry_point: "main",
                    },
                }),
            ));
        }

        let staging = device
            .create_buffer_with_data(bytemuck::bytes_of(uniforms), wgpu::BufferUsage::COPY_SRC);
        encoder.copy_buffer_to_buffer(&staging, 0, &self.uniforms, 0, mem::size_of::<U>() as u64);

        let mut cpass = encoder.begin_compute_pass();
        cpass.set_pipeline(&self.bindgroup_pipeline.as_ref().unwrap().1);
        cpass.set_bind_group(0, &self.bindgroup_pipeline.as_ref().unwrap().0, &[]);
        cpass.dispatch(dimensions.0, dimensions.1, dimensions.2);
    }
}
