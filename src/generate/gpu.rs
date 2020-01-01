use crate::cache::{AssetLoadContext, MMappedAsset, WebAsset};
use crate::coordinates::{CoordinateSystem, PLANET_RADIUS};
use crate::terrain::dem::{DemSource, GlobalDem};
use crate::terrain::raster::{RasterCache, RasterSource};
use crate::GpuState;
use cgmath::InnerSpace;
use std::mem;
use std::rc::Rc;

const BASE_RESOLUTION: u32 = 1024;

#[derive(Copy, Clone)]
pub(crate) struct GenHeightsUniforms {
	position: [f32; 2],
}
unsafe impl bytemuck::Zeroable for GenHeightsUniforms {}
unsafe impl bytemuck::Pod for GenHeightsUniforms {}

pub(crate) struct ComputeShader<U> {
    shader: rshader::ShaderSet,
    bind_group: wgpu::BindGroup,
    compute_pipeline_layout: wgpu::PipelineLayout,
    compute_pipeline: Option<wgpu::ComputePipeline>,

    uniforms: wgpu::Buffer,
    _phantom: std::marker::PhantomData<U>,
}
impl<U: bytemuck::Pod> ComputeShader<U> {
    pub fn new(
        device: &wgpu::Device,
        queue: &mut wgpu::Queue,
        state: &GpuState,
        shader: rshader::ShaderSet,
    ) -> Self {
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            size: mem::size_of::<U>() as u64,
            usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::UNIFORM,
        });

        let texture_binding = wgpu::BindGroupLayoutBinding {
            binding: 0,
            visibility: wgpu::ShaderStage::COMPUTE,
            ty: wgpu::BindingType::SampledTexture {
                multisampled: false,
                dimension: wgpu::TextureViewDimension::D2,
            },
        };

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            bindings: &[
                wgpu::BindGroupLayoutBinding {
                    binding: 0,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::UniformBuffer { dynamic: false },
                },
                wgpu::BindGroupLayoutBinding { binding: 1, ..texture_binding },
                wgpu::BindGroupLayoutBinding { binding: 2, ..texture_binding },
                wgpu::BindGroupLayoutBinding {
                    binding: 3,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::Sampler,
                },
                wgpu::BindGroupLayoutBinding { binding: 4, ..texture_binding },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            bindings: &[
                wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &uniforms,
                        range: 0..mem::size_of::<U>() as u64,
                    },
                },
                wgpu::Binding {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &state.base_heights.create_default_view(),
                    ),
                },
                wgpu::Binding {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        &state.heights_staging.create_default_view(),
                    ),
                },
                wgpu::Binding {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&device.create_sampler(
                        &wgpu::SamplerDescriptor {
                            address_mode_u: wgpu::AddressMode::ClampToEdge,
                            address_mode_v: wgpu::AddressMode::ClampToEdge,
                            address_mode_w: wgpu::AddressMode::ClampToEdge,
                            mag_filter: wgpu::FilterMode::Linear,
                            min_filter: wgpu::FilterMode::Linear,
                            mipmap_filter: wgpu::FilterMode::Nearest,
                            lod_min_clamp: -100.0,
                            lod_max_clamp: 100.0,
                            compare_function: wgpu::CompareFunction::Always,
                        },
                    )),
                },
                wgpu::Binding {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(
                        &state.noise.create_default_view(),
                    ),
                },
            ],
        });
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[&bind_group_layout],
            });

        Self {
            shader,
            bind_group,
            compute_pipeline_layout,
            compute_pipeline: None,
            uniforms,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn refresh(&mut self, watcher: &mut rshader::ShaderDirectoryWatcher) {
        if self.shader.refresh(watcher) {
            self.compute_pipeline = None;
        }
    }

    pub fn run(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        dimensions: (u32, u32, u32),
        uniforms: &U,
    ) {
        if self.compute_pipeline.is_none() {
            self.compute_pipeline =
                Some(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    layout: &self.compute_pipeline_layout,
                    compute_stage: wgpu::ProgrammableStageDescriptor {
                        module: &device.create_shader_module(
                            &wgpu::read_spirv(std::io::Cursor::new(self.shader.compute())).unwrap(),
                        ),
                        entry_point: "main",
                    },
                }));
        }

        let staging = device
            .create_buffer_with_data(bytemuck::bytes_of(uniforms), wgpu::BufferUsage::COPY_SRC);
        encoder.copy_buffer_to_buffer(&staging, 0, &self.uniforms, 0, mem::size_of::<U>() as u64);

        let mut cpass = encoder.begin_compute_pass();
        cpass.set_pipeline(self.compute_pipeline.as_ref().unwrap());
        cpass.set_bind_group(0, &self.bind_group, &[]);
        cpass.dispatch(dimensions.0, dimensions.1, dimensions.2);
    }
}
