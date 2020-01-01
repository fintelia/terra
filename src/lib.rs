//! Terra is a large scale terrain generation and rendering library built on top of wgpu.
#![feature(stmt_expr_attributes)]
#![feature(with_options)]

#[macro_use]
extern crate lazy_static;
extern crate rshader;

mod coordinates;
mod generate;
// mod graph;
mod cache;
mod mapfile;
mod runtime_texture;
mod srgb;
mod terrain;
mod utils;

// pub mod plugin;
// pub mod compute;

use crate::generate::{ComputeShader, GenHeightsUniforms};
use crate::mapfile::MapFile;
use crate::terrain::quadtree::render::NodeState;
use crate::terrain::tile_cache::{LayerType, TileCache};
use std::mem;
use terrain::quadtree::QuadTree;
use vec_map::VecMap;
use wgpu_glyph::{GlyphBrush, Section};

pub use generate::{GridSpacing, QuadTreeBuilder, TextureQuality, VertexQuality};

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

    noise: wgpu::Texture,
    planet_mesh_texture: wgpu::Texture,

    heights: wgpu::Texture,
    normals: wgpu::Texture,
    albedo: wgpu::Texture,
}

pub struct Terrain {
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,

    render_pipeline_layout: wgpu::PipelineLayout,
    render_pipeline: Option<wgpu::RenderPipeline>,

    watcher: rshader::ShaderDirectoryWatcher,
    shader: rshader::ShaderSet,

    uniform_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_buffer_partial: wgpu::Buffer,

    gen_heights: ComputeShader<GenHeightsUniforms>,
    // gen_normals: ComputeShader<()>,
    // load_heights: ComputeShader<()>,
    gpu_state: GpuState,
    quadtree: QuadTree,
    mapfile: MapFile,
    tile_cache: VecMap<TileCache>,

    glyph_brush: GlyphBrush<'static, ()>,
}
impl Terrain {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &mut wgpu::Queue,
        mut mapfile: MapFile,
    ) -> Self {
        let mut tile_cache = VecMap::new();
        for layer in mapfile.layers().values().cloned() {
            tile_cache
                .insert(layer.layer_type as usize, TileCache::new(layer));
        }

        let quadtree = QuadTree::new(
            mapfile.take_nodes(),
            tile_cache[LayerType::Heights.index()].resolution() - 1,
        );

        let mut watcher = rshader::ShaderDirectoryWatcher::new("src/shaders").unwrap();
        let shader = rshader::ShaderSet::simple(
            &mut watcher,
            rshader::shader_source!("shaders", "version", "a.vert"),
            rshader::shader_source!("shaders", "version", "pbr", "a.frag"),
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

        let heights_resolution = tile_cache[LayerType::Heights.index()].resolution();
        let normals_resolution = tile_cache[LayerType::Normals.index()].resolution();
        let albedo_resolution = tile_cache[LayerType::Colors.index()].resolution();

        let (noise, planet_mesh_texture, base_heights) = {
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
            let noise = mapfile.noise_texture(device, &mut encoder);
            let planet_mesh_texture = mapfile.planet_mesh_texture(device, &mut encoder);
			let base_heights = mapfile.base_heights_texture(device, &mut encoder);
            queue.submit(&[encoder.finish()]);
            (noise, planet_mesh_texture, base_heights)
        };

        let gpu_state = GpuState {
            noise,
            planet_mesh_texture,
            base_heights,
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
            heights: tile_cache[LayerType::Heights].make_cache_texture(device),
            normals: tile_cache[LayerType::Normals].make_cache_texture(device),
            albedo: tile_cache[LayerType::Colors].make_cache_texture(device),
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
                        &gpu_state.albedo.create_default_view(),
                    ),
                },
            ],
        });
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[&bind_group_layout],
            });

        let gen_heights = ComputeShader::new(
            device,
            queue,
            &gpu_state,
            rshader::ShaderSet::compute_only(
                &mut watcher,
                rshader::shader_source!("shaders", "version", "gen-heights.comp"),
            )
            .unwrap(),
        );

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

        let font: &'static [u8] = include_bytes!("../assets/UbuntuMono/UbuntuMono-R.ttf");
        let glyph_brush = wgpu_glyph::GlyphBrushBuilder::using_font_bytes(font)
            .build(&device, wgpu::TextureFormat::Bgra8UnormSrgb);

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

			gen_heights,

            gpu_state,
            // gen_heights,
            // gen_normals,
            // load_heights,
            quadtree,
            mapfile,
            tile_cache,
            glyph_brush,
        }
    }

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &mut wgpu::Queue,
        frame: &wgpu::SwapChainOutput,
        depth_buffer: &wgpu::TextureView,
        frame_size: (u32, u32),
        view_proj: mint::ColumnMatrix4<f32>,
        camera: mint::Point3<f32>,
    ) {
        let camera_frustum = collision::Frustum::from_matrix4(view_proj.into());

        self.quadtree.update(&mut self.tile_cache, camera, camera_frustum);
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
                        cull_mode: wgpu::CullMode::Front,
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
                                format: wgpu::VertexFormat::Float4,
                                shader_location: 2,
                            },
                            wgpu::VertexAttributeDescriptor {
                                offset: 48,
                                format: wgpu::VertexFormat::Float4,
                                shader_location: 3,
                            },
                            wgpu::VertexAttributeDescriptor {
                                offset: 64,
                                format: wgpu::VertexFormat::Float4,
                                shader_location: 4,
                            },
                            wgpu::VertexAttributeDescriptor {
                                offset: 80,
                                format: wgpu::VertexFormat::Float4,
                                shader_location: 5,
                            },
                            wgpu::VertexAttributeDescriptor {
                                offset: 96,
                                format: wgpu::VertexFormat::Int,
                                shader_location: 6,
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

        self.tile_cache[LayerType::Heights.index()].upload_tiles(
            device,
            &mut encoder,
            &self.gpu_state.heights,
			&self.quadtree.nodes,
			&mut self.mapfile,
        );
        self.tile_cache[LayerType::Normals.index()].upload_tiles(
            device,
            &mut encoder,
            &self.gpu_state.normals,
			&self.quadtree.nodes,
			&mut self.mapfile,
        );
        self.tile_cache[LayerType::Colors.index()].upload_tiles(
            device,
            &mut encoder,
            &self.gpu_state.albedo,
			&self.quadtree.nodes,
			&mut self.mapfile,
        );

        self.quadtree.prepare_vertex_buffer(
            device,
            &mut encoder,
            &mut self.vertex_buffer,
            &self.tile_cache,
        );

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
                    clear_color: wgpu::Color { r: 0.0, g: 0.3, b: 0.8, a: 1.0 },
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

        {
            use std::fmt::Write;
            let mut text = String::new();
            let heights = &self.tile_cache[LayerType::Heights.index()];
            writeln!(
                &mut text,
                "tile_cache[heights] = {} / {}",
                heights.utilization(),
                heights.capacity()
            )
            .unwrap();

            let normals = &self.tile_cache[LayerType::Normals.index()];
            writeln!(
                &mut text,
                "tile_cache[normals] = {} / {}",
                normals.utilization(),
                normals.capacity()
            )
            .unwrap();

            let albedo = &self.tile_cache[LayerType::Colors.index()];
            writeln!(
                &mut text,
                "tile_cache[albedo] = {} / {}",
                albedo.utilization(),
                albedo.capacity()
            )
            .unwrap();

            writeln!(&mut text, "x = {}\ny = {}\nz = {}", camera.x, camera.y, camera.z).unwrap();

            self.glyph_brush.queue(Section {
                text: &text,
                screen_position: (10.0, 10.0),
                ..Default::default()
            });
        }

        self.glyph_brush
            .draw_queued(device, &mut encoder, &frame.view, frame_size.0, frame_size.1)
            .unwrap();
        queue.submit(&[encoder.finish()]);
    }
}
