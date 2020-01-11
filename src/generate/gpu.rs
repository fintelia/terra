use crate::GpuState;
use std::mem;

#[derive(Copy, Clone)]
pub(crate) struct GenHeightsUniforms {
    pub position: [f32; 2],
    pub base_heights_step: f32,
    pub step: f32,
}
unsafe impl bytemuck::Zeroable for GenHeightsUniforms {}
unsafe impl bytemuck::Pod for GenHeightsUniforms {}

#[derive(Copy, Clone)]
pub(crate) struct GenNormalsUniforms {
    pub position: [f32; 2],
    pub spacing: f32,
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
            self.regenerate_shader(device, state);
        }

        let staging = device
            .create_buffer_with_data(bytemuck::bytes_of(uniforms), wgpu::BufferUsage::COPY_SRC);
        encoder.copy_buffer_to_buffer(&staging, 0, &self.uniforms, 0, mem::size_of::<U>() as u64);

        let mut cpass = encoder.begin_compute_pass();
        cpass.set_pipeline(&self.bindgroup_pipeline.as_ref().unwrap().1);
        cpass.set_bind_group(0, &self.bindgroup_pipeline.as_ref().unwrap().0, &[]);
        cpass.dispatch(dimensions.0, dimensions.1, dimensions.2);
    }

    fn regenerate_shader(&mut self, device: &wgpu::Device, state: &GpuState) {
        let linear = &device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: -100.0,
            lod_max_clamp: 100.0,
            compare_function: wgpu::CompareFunction::Always,
        });
        let linear_wrap = &device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: -100.0,
            lod_max_clamp: 100.0,
            compare_function: wgpu::CompareFunction::Always,
        });

        let base_heights = &state.base_heights.create_default_view();
        let heights_staging = &state.heights_staging.create_default_view();
        let normals_staging = &state.normals_staging.create_default_view();
        let noise = &state.noise.create_default_view();

        let bind_group_layout =
            device.create_bind_group_layout(&self.shader.layout_descriptor().unwrap());
        let mut bindings = Vec::new();
        for (name, layout) in self
            .shader
            .desc_names()
            .unwrap()
            .iter()
            .zip(self.shader.layout_descriptor().unwrap().bindings.iter())
        {
            let name = &**name.as_ref().unwrap();
            bindings.push(wgpu::Binding {
                binding: layout.binding,
                resource: match layout.ty {
                    wgpu::BindingType::Sampler => wgpu::BindingResource::Sampler(match name {
                        "linear" => &linear,
                        "linear_wrap" => &linear_wrap,
                        _ => unreachable!("unrecognized sampler: {}", name),
                    }),
                    wgpu::BindingType::UniformBuffer { .. } => wgpu::BindingResource::Buffer {
                        buffer: &self.uniforms,
                        range: 0..mem::size_of::<U>() as u64,
                    },
                    wgpu::BindingType::StorageTexture { .. }
                    | wgpu::BindingType::SampledTexture { .. } => {
                        wgpu::BindingResource::TextureView(match name {
                            "base_heights" => base_heights,
                            "heights_staging" => heights_staging,
                            "normals_staging" => normals_staging,
                            "noise" => noise,
                            _ => unreachable!("unrecognized image: {}", name),
                        })
                    }
                    wgpu::BindingType::StorageBuffer { .. } => unimplemented!(),
                },
            })
        }

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            bindings: &*bindings,
        });

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
}
