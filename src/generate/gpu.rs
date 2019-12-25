use crate::cache::{AssetLoadContext, MMappedAsset, WebAsset};
use crate::coordinates::{CoordinateSystem, PLANET_RADIUS};
use crate::terrain::dem::{DemSource, GlobalDem};
use crate::terrain::raster::{RasterCache, RasterSource};
use crate::GpuState;
use cgmath::InnerSpace;
use std::mem;
use std::rc::Rc;

const BASE_RESOLUTION: u32 = 1024;

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

        let storage_texture_binding = wgpu::BindGroupLayoutBinding {
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
                    ty: wgpu::BindingType::SampledTexture {
                        multisampled: false,
                        dimension: wgpu::TextureViewDimension::D2,
                    },
                },
                wgpu::BindGroupLayoutBinding {
                    binding: 1,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::Sampler,
                },
                wgpu::BindGroupLayoutBinding { binding: 2, ..storage_texture_binding },
                wgpu::BindGroupLayoutBinding { binding: 3, ..storage_texture_binding },
                wgpu::BindGroupLayoutBinding { binding: 4, ..storage_texture_binding },
                wgpu::BindGroupLayoutBinding { binding: 5, ..storage_texture_binding },
                wgpu::BindGroupLayoutBinding {
                    binding: 6,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::UniformBuffer { dynamic: false },
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            bindings: &[
                wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &state.base_heights.create_default_view(),
                    ),
                },
                wgpu::Binding {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&device.create_sampler(
                        &wgpu::SamplerDescriptor {
                            address_mode_u: wgpu::AddressMode::ClampToEdge,
                            address_mode_v: wgpu::AddressMode::ClampToEdge,
                            address_mode_w: wgpu::AddressMode::ClampToEdge,
                            mag_filter: wgpu::FilterMode::Linear,
                            min_filter: wgpu::FilterMode::Linear,
                            mipmap_filter: wgpu::FilterMode::Nearest,
                            lod_min_clamp: 0.0,
                            lod_max_clamp: 0.0,
                            compare_function: wgpu::CompareFunction::Never,
                        },
                    )),
                },
                wgpu::Binding {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        &state.heights_staging.create_default_view(),
                    ),
                },
                wgpu::Binding {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(
                        &state.normals_staging.create_default_view(),
                    ),
                },
                wgpu::Binding {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(
                        &state.heights.create_default_view(),
                    ),
                },
                wgpu::Binding {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(
                        &state.normals.create_default_view(),
                    ),
                },
                wgpu::Binding {
                    binding: 6,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &uniforms,
                        range: 0..mem::size_of::<U>() as u64,
                    },
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

fn world_position(x: u32, y: u32) -> cgmath::Vector2<f64> {
    let fx = x as f32 / (BASE_RESOLUTION - 1) as f32;
    let fy = y as f32 / (BASE_RESOLUTION - 1) as f32;

    let min = -32.0 * BASE_RESOLUTION as f32;
    let max = 32.0 * BASE_RESOLUTION as f32;

    cgmath::Vector2::new((min + (max - min) * fx) as f64, (min + (max - min) * fy) as f64)
}

pub fn make_base_heights(
    device: &wgpu::Device,
    queue: &mut wgpu::Queue,
    system: &CoordinateSystem,
    base_heights: &wgpu::Texture,
) {
    let mut context = AssetLoadContext::new();
    // let global = GlobalDem.load(&mut context);

    let mut dem_cache: RasterCache<f32, Vec<f32>> =
        RasterCache::new(Box::new(DemSource::Usgs30m), 64);

    let staging_buffer = device.create_buffer_mapped(
        (BASE_RESOLUTION * BASE_RESOLUTION * 4) as usize,
        wgpu::BufferUsage::COPY_SRC | wgpu::BufferUsage::MAP_WRITE,
    );
    let data = bytemuck::cast_slice_mut(staging_buffer.data);
    for y in 0..BASE_RESOLUTION {
        for x in 0..BASE_RESOLUTION {
            let world = world_position(x, y);
            let mut world3 = cgmath::Vector3::new(
                world.x,
                PLANET_RADIUS * ((1.0 - world.magnitude2() / PLANET_RADIUS).max(0.25).sqrt() - 1.0),
                world.y,
            );
            for i in 0..5 {
                world3.x = world.x;
                world3.z = world.y;
                let mut lla = system.world_to_lla(world3);
                lla.z = dem_cache
                    .interpolate(&mut context, lla.x.to_degrees(), lla.y.to_degrees(), 0)
                    .unwrap_or(0.0) as f64;
                world3 = system.lla_to_world(lla);
            }
            data[(x + y * BASE_RESOLUTION) as usize] = world3.y as f32;
        }
    }
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
    encoder.copy_buffer_to_texture(
        wgpu::BufferCopyView {
            buffer: &staging_buffer.finish(),
            offset: 0,
            row_pitch: BASE_RESOLUTION * 4,
            image_height: BASE_RESOLUTION,
        },
        wgpu::TextureCopyView {
            texture: &base_heights,
            mip_level: 0,
            array_layer: 0,
            origin: wgpu::Origin3d { x: 0.0, y: 0.0, z: 0.0 },
        },
        wgpu::Extent3d { width: BASE_RESOLUTION, height: BASE_RESOLUTION, depth: 1 },
    );
    queue.submit(&[encoder.finish()]);
}
