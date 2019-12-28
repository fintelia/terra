//! Terra is a large scale terrain generation and rendering library built on top of wgpu.
#![feature(custom_attribute)]
#![feature(stmt_expr_attributes)]
#![feature(float_to_from_bytes)]

#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate rshader;

mod coordinates;
mod generate;
// mod graph;
mod cache;
mod runtime_texture;
mod srgb;
mod terrain;
mod utils;

// pub mod plugin;
// pub mod compute;

use crate::generate::ComputeShader;
use crate::terrain::quadtree::render::NodeState;
pub use generate::{GridSpacing, QuadTreeBuilder, TextureQuality, VertexQuality};
use std::mem;
pub use terrain::quadtree::QuadTree;

#[repr(C)]
#[derive(Copy, Clone)]
struct UniformBlock {
    view_proj: mint::ColumnMatrix4<f32>,
    camera: mint::Point3<f32>,
    padding: f32,
}
unsafe impl bytemuck::Pod for UniformBlock {}
unsafe impl bytemuck::Zeroable for UniformBlock {}

pub(crate) struct GpuState {
    base_heights: wgpu::Texture,

    heights_staging: wgpu::Texture,
    normals_staging: wgpu::Texture,

    heights: wgpu::Texture,
    normals: wgpu::Texture,
    albedo: wgpu::Texture,
}

pub struct Terrain {
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,

    render_pipeline_layout: wgpu::PipelineLayout,
    render_pipeline: Option<wgpu::RenderPipeline>,

    // set_layouts: Vec<Handle<DescriptorSetLayout<B>>>,
    // descriptor_sets: Vec<Escape<DescriptorSet<B>>>,
    // pipeline_layout: B::PipelineLayout,
    // graphics_pipeline: B::GraphicsPipeline,
    watcher: rshader::ShaderDirectoryWatcher,
    shader: rshader::ShaderSet,

    uniform_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_buffer_partial: wgpu::Buffer,

    // gen: TileGen,
    gpu_state: GpuState,

    // gen_heights: ComputeShader<()>,
    // gen_normals: ComputeShader<()>,
    // load_heights: ComputeShader<()>,

    // heightmap: wgpu::Texture,
    quadtree: QuadTree,
}
impl Terrain {
    pub fn new(device: &wgpu::Device, queue: &mut wgpu::Queue, quadtree: QuadTree) -> Self {
        let mut watcher = rshader::ShaderDirectoryWatcher::new("src/shaders").unwrap();
        let shader = rshader::ShaderSet::simple(
            &mut watcher,
            rshader::shader_source!("shaders", "version", "a.vert"),
            rshader::shader_source!("shaders", "version", "a.frag"),
        )
        .unwrap();

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            size: std::mem::size_of::<UniformBlock>() as u64,
            usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::UNIFORM,
        });
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            size: (std::mem::size_of::<NodeState>() * quadtree.total_nodes()) as u64,
            usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::VERTEX,
        });
        let (index_buffer, index_buffer_partial) = quadtree.create_index_buffers(device);

        let texture_desc = wgpu::TextureDescriptor {
            size: wgpu::Extent3d { width: 0, height: 0, depth: 0 },
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsage::COPY_SRC
                | wgpu::TextureUsage::COPY_DST
                | wgpu::TextureUsage::SAMPLED
                | wgpu::TextureUsage::STORAGE,
        };

        use crate::terrain::tile_cache::LayerType;
        let heights_resolution =
            quadtree.tile_cache_layers[LayerType::Heights.index()].resolution();
        let normals_resolution =
            quadtree.tile_cache_layers[LayerType::Normals.index()].resolution();
        let albedo_resolution =
            quadtree.tile_cache_layers[LayerType::Colors.index()].resolution();

        let gpu_state = GpuState {
            base_heights: device.create_texture(&wgpu::TextureDescriptor {
                size: wgpu::Extent3d { width: 1024, height: 1024, depth: 1 },
                format: wgpu::TextureFormat::R32Float,
                ..texture_desc
            }),
            heights_staging: device.create_texture(&wgpu::TextureDescriptor {
                size: wgpu::Extent3d { width: 1025, height: 1025, depth: 1 },
                format: wgpu::TextureFormat::R32Float,
                ..texture_desc
            }),
            normals_staging: device.create_texture(&wgpu::TextureDescriptor {
                size: wgpu::Extent3d { width: 1024, height: 1024, depth: 1 },
                format: wgpu::TextureFormat::Rg8Uint,
                ..texture_desc
            }),
            heights: device.create_texture(&wgpu::TextureDescriptor {
                size: wgpu::Extent3d {
                    width: heights_resolution,
                    height: heights_resolution,
                    depth: 1,
                },
                format: wgpu::TextureFormat::R32Float,
                array_layer_count: 512,
                ..texture_desc
            }),
            normals: device.create_texture(&wgpu::TextureDescriptor {
                size: wgpu::Extent3d {
                    width: normals_resolution,
                    height: normals_resolution,
                    depth: 1,
                },
                format: wgpu::TextureFormat::Rgba8Unorm,
                array_layer_count: 256,
                ..texture_desc
            }),
            albedo: device.create_texture(&wgpu::TextureDescriptor {
                size: wgpu::Extent3d {
                    width: albedo_resolution,
                    height: albedo_resolution,
                    depth: 1,
                },
                format: wgpu::TextureFormat::Rgba8Unorm,
                array_layer_count: 256,
                ..texture_desc
            }),
        };

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            bindings: &[
                wgpu::BindGroupLayoutBinding {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::UniformBuffer { dynamic: false },
                },
                wgpu::BindGroupLayoutBinding {
                    binding: 1,
                    visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler,
                },
                wgpu::BindGroupLayoutBinding {
                    binding: 2,
                    visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::SampledTexture {
                        multisampled: false,
                        dimension: wgpu::TextureViewDimension::D2Array,
                    },
                },
                wgpu::BindGroupLayoutBinding {
                    binding: 3,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::SampledTexture {
                        multisampled: false,
                        dimension: wgpu::TextureViewDimension::D2Array,
                    },
                },
                wgpu::BindGroupLayoutBinding {
                    binding: 4,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::SampledTexture {
                        multisampled: false,
                        dimension: wgpu::TextureViewDimension::D2Array,
                    },
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            bindings: &[
                wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &uniform_buffer,
                        range: 0..mem::size_of::<UniformBlock>() as u64,
                    },
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
                            lod_min_clamp: -100.0,
                            lod_max_clamp: 100.0,
                            compare_function: wgpu::CompareFunction::Always,
                        },
                    )),
                },
                wgpu::Binding {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        &gpu_state.heights.create_default_view(),
                    ),
                },
                wgpu::Binding {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(
                        &gpu_state.normals.create_default_view(),
                    ),
                },
                wgpu::Binding {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(
                        &gpu_state.normals.create_default_view(),
                    ),
                },
            ],
        });
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[&bind_group_layout],
            });

        crate::generate::make_base_heights(
            device,
            queue,
            &quadtree.system,
            &gpu_state.base_heights,
        );

        // let gen_heights = ComputeShader::new(
        //     device,
        //     queue,
        //     &gpu_state,
        //     rshader::ShaderSet::compute_only(
        //         &mut watcher,
        //         rshader::shader_source!("shaders", "version", "gen-heights.comp"),
        //     )
        //     .unwrap(),
        // );

        // let gen_normals = ComputeShader::new(
        //     device,
        //     queue,
        //     &gpu_state,
        //     rshader::ShaderSet::compute_only(
        //         &mut watcher,
        //         rshader::shader_source!("shaders", "version", "gen-normals.comp"),
        //     )
        //     .unwrap(),
        // );

        // let load_heights = ComputeShader::new(
        //     device,
        //     queue,
        //     &gpu_state,
        //     rshader::ShaderSet::compute_only(
        //         &mut watcher,
        //         rshader::shader_source!("shaders", "version", "load-heights.comp"),
        //     )
        //     .unwrap(),
        // );

        Self {
            bind_group_layout,
            bind_group,
            render_pipeline_layout,
            render_pipeline: None,
            watcher,
            shader,

            uniform_buffer,
            vertex_buffer,
            index_buffer,
            index_buffer_partial,

            gpu_state,
            // gen_heights,
            // gen_normals,
            // load_heights,
            quadtree,
        }
    }

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &mut wgpu::Queue,
        frame: &wgpu::SwapChainOutput,
        depth_buffer: &wgpu::TextureView,
        view_proj: mint::ColumnMatrix4<f32>,
        camera: mint::Point3<f32>,
    ) {
        let camera_frustum = collision::Frustum::from_matrix4(view_proj.into());

        self.quadtree.update(camera, camera_frustum);
        if self.shader.refresh(&mut self.watcher) {
            self.render_pipeline = None;
        }

        if self.render_pipeline.is_none() {
            self.render_pipeline = Some(
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    layout: &self.render_pipeline_layout,
                    vertex_stage: wgpu::ProgrammableStageDescriptor {
                        module: &device.create_shader_module(
                            &wgpu::read_spirv(std::io::Cursor::new(self.shader.vertex())).unwrap(),
                        ),
                        entry_point: "main",
                    },
                    fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                        module: &device.create_shader_module(
                            &wgpu::read_spirv(std::io::Cursor::new(self.shader.fragment()))
                                .unwrap(),
                        ),
                        entry_point: "main",
                    }),
                    rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: wgpu::CullMode::None,
                        depth_bias: 0,
                        depth_bias_slope_scale: 0.0,
                        depth_bias_clamp: 0.0,
                    }),
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
                        stencil_front: Default::default(),
                        stencil_back: Default::default(),
                        stencil_read_mask: 0,
                        stencil_write_mask: 0,
                    }),
                    index_format: wgpu::IndexFormat::Uint16,
                    vertex_buffers: &[wgpu::VertexBufferDescriptor {
                        stride: std::mem::size_of::<NodeState>() as u64,
                        step_mode: wgpu::InputStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttributeDescriptor {
                                offset: 0,
                                format: wgpu::VertexFormat::Float4,
                                shader_location: 0,
                            },
                            wgpu::VertexAttributeDescriptor {
                                offset: 16,
                                format: wgpu::VertexFormat::Float4,
                                shader_location: 1,
                            },
                            wgpu::VertexAttributeDescriptor {
                                offset: 32,
                                format: wgpu::VertexFormat::Float2,
                                shader_location: 4,
                            },
                            wgpu::VertexAttributeDescriptor {
                                offset: 40,
                                format: wgpu::VertexFormat::Float2,
                                shader_location: 5,
                            },
                            wgpu::VertexAttributeDescriptor {
                                offset: 48,
                                format: wgpu::VertexFormat::Float2,
                                shader_location: 6,
                            },
                            wgpu::VertexAttributeDescriptor {
                                offset: 56,
                                format: wgpu::VertexFormat::Float2,
                                shader_location: 7,
                            },
                            wgpu::VertexAttributeDescriptor {
                                offset: 64,
                                format: wgpu::VertexFormat::Float2,
                                shader_location: 8,
                            },
                            wgpu::VertexAttributeDescriptor {
                                offset: 72,
                                format: wgpu::VertexFormat::Float,
                                shader_location: 9,
                            },
                            wgpu::VertexAttributeDescriptor {
                                offset: 76,
                                format: wgpu::VertexFormat::Float,
                                shader_location: 10,
                            },
                            wgpu::VertexAttributeDescriptor {
                                offset: 80,
                                format: wgpu::VertexFormat::Int,
                                shader_location: 11,
                            },
                        ],
                    }],
                    sample_count: 1,
                    sample_mask: !0,
                    alpha_to_coverage_enabled: false,
                }),
            );
        }

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });

        self.quadtree.upload_tiles(device, &mut encoder, &self.gpu_state);
        self.quadtree.prepare_vertex_buffer(device, &mut encoder, &mut self.vertex_buffer);

        let mapped = device.create_buffer_mapped(
            mem::size_of::<UniformBlock>(),
            wgpu::BufferUsage::MAP_WRITE | wgpu::BufferUsage::COPY_SRC,
        );
        bytemuck::cast_slice_mut(mapped.data)[0] = UniformBlock { view_proj, camera, padding: 0.0 };
        encoder.copy_buffer_to_buffer(
            &mapped.finish(),
            0,
            &self.uniform_buffer,
            0,
            mem::size_of::<UniformBlock>() as u64,
        );

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &frame.view,
                    resolve_target: None,
                    load_op: wgpu::LoadOp::Clear,
                    store_op: wgpu::StoreOp::Store,
                    clear_color: wgpu::Color::BLACK,
                }],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                    attachment: depth_buffer,
                    depth_load_op: wgpu::LoadOp::Clear,
                    depth_store_op: wgpu::StoreOp::Store,
                    clear_depth: 0.0,
                    stencil_load_op: wgpu::LoadOp::Clear,
                    stencil_store_op: wgpu::StoreOp::Store,
                    clear_stencil: 0,
                }),
            });
            rpass.set_pipeline(self.render_pipeline.as_ref().unwrap());
            rpass.set_bind_group(0, &self.bind_group, &[]);

            self.quadtree.render(
                &mut rpass,
                &self.vertex_buffer,
                &self.index_buffer,
                &self.index_buffer_partial,
            );
        }

        queue.submit(&[encoder.finish()]);
    }
}
