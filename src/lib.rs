//! Terra is a large scale terrain generation and rendering library built on top of wgpu.
#![feature(async_closure)]
#![feature(stmt_expr_attributes)]
#![feature(with_options)]

#[macro_use]
extern crate lazy_static;
extern crate rshader;

mod cache;
mod coordinates;
mod generate;
mod gpu_state;
mod mapfile;
mod srgb;
mod terrain;
mod utils;

use crate::generate::{ComputeShader, GenHeightsUniforms, GenNormalsUniforms};
use crate::terrain::quadtree::render::NodeState;
use crate::terrain::tile_cache::{LayerType, TileCache};
use futures::executor;
use gpu_state::GpuState;
use std::mem;
use terrain::quadtree::QuadTree;
use vec_map::VecMap;
use wgpu_glyph::{GlyphBrush, Section};

pub use crate::mapfile::MapFile;
pub use generate::{GridSpacing, MapFileBuilder, TextureQuality, VertexQuality};

#[repr(C)]
#[derive(Copy, Clone)]
struct UniformBlock {
    view_proj: mint::ColumnMatrix4<f32>,
    camera: mint::Point3<f32>,
    padding: f32,
}
unsafe impl bytemuck::Pod for UniformBlock {}
unsafe impl bytemuck::Zeroable for UniformBlock {}

pub struct Terrain {
    _bind_group_layout: wgpu::BindGroupLayout,
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
    gen_normals: ComputeShader<GenNormalsUniforms>,
    gpu_state: GpuState,
    quadtree: QuadTree,
    mapfile: MapFile,
    tile_cache: VecMap<TileCache>,

    pending_tiles: Vec<(LayerType, usize, wgpu::Buffer)>,

    glyph_brush: GlyphBrush<'static, ()>,
}
impl Terrain {
    pub fn new(device: &wgpu::Device, queue: &mut wgpu::Queue, mut mapfile: MapFile) -> Self {
        let mut tile_cache = VecMap::new();
        for layer in mapfile.layers().values().cloned() {
            tile_cache.insert(layer.layer_type as usize, TileCache::new(layer));
        }

        let quadtree = QuadTree::new(tile_cache[LayerType::Heights.index()].resolution() - 1);

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
            size: (std::mem::size_of::<NodeState>() * 1024) as u64,
            usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::VERTEX,
        });
        let (index_buffer, index_buffer_partial) = quadtree.create_index_buffers(device);

        let staging_texture_desc = wgpu::TextureDescriptor {
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
            _planet_mesh_texture: planet_mesh_texture,
            base_heights,
            heights_staging: device.create_texture(&wgpu::TextureDescriptor {
                size: wgpu::Extent3d { width: 1025, height: 1025, depth: 1 },
                format: wgpu::TextureFormat::Rgba32Float,
                ..staging_texture_desc
            }),
            normals_staging: device.create_texture(&wgpu::TextureDescriptor {
                size: wgpu::Extent3d { width: 1024, height: 1024, depth: 1 },
                format: wgpu::TextureFormat::Rg8Unorm,
                ..staging_texture_desc
            }),
            heights: tile_cache[LayerType::Heights].make_cache_texture(device),
            normals: tile_cache[LayerType::Normals].make_cache_texture(device),
            albedo: tile_cache[LayerType::Colors].make_cache_texture(device),
        };

        let (bind_group, bind_group_layout) = gpu_state.bind_group_for_shader(
            device,
            &shader,
            Some(&wgpu::BindingResource::Buffer {
                buffer: &uniform_buffer,
                range: 0..mem::size_of::<UniformBlock>() as u64,
            }),
        );
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[&bind_group_layout],
            });

        let gen_heights = ComputeShader::new(
            device,
            rshader::ShaderSet::compute_only(
                &mut watcher,
                rshader::shader_source!("shaders", "version", "gen-heights.comp"),
            )
            .unwrap(),
        );
        let gen_normals = ComputeShader::new(
            device,
            rshader::ShaderSet::compute_only(
                &mut watcher,
                rshader::shader_source!("shaders", "version", "gen-normals.comp"),
            )
            .unwrap(),
        );

        // TODO: only clear if shader has changed?
        mapfile.clear_generated(LayerType::Heights);
        mapfile.clear_generated(LayerType::Normals);

        let font: &'static [u8] = include_bytes!("../assets/UbuntuMono/UbuntuMono-R.ttf");
        let glyph_brush = wgpu_glyph::GlyphBrushBuilder::using_font_bytes(font)
            .build(device, wgpu::TextureFormat::Bgra8UnormSrgb);

        Self {
            _bind_group_layout: bind_group_layout,
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
            gen_normals,

            gpu_state,
            quadtree,
            mapfile,
            tile_cache,
            glyph_brush,

            pending_tiles: Vec::new(),
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
        for (layer, tile, buffer) in self.pending_tiles.drain(..) {
            let resolution = self.tile_cache[layer.index()].resolution() as usize;
            let bytes_per_texel = self.tile_cache[layer.index()].bytes_per_texel();
            let row_pitch = self.tile_cache[layer.index()].row_pitch();
            let row_bytes = resolution * bytes_per_texel;
            let size = row_pitch * resolution;

            let future = buffer.map_read(0, size as u64);

            device.poll(false);

            if let Ok(mapping) = executor::block_on(future) {
                let mut tile_data = vec![0; resolution * resolution * bytes_per_texel];
                let buffer_data = mapping.as_slice();
                for row in 0..resolution {
                    tile_data[row * row_bytes..][..row_bytes]
                        .copy_from_slice(&buffer_data[row * row_pitch..][..row_bytes]);
                }
                self.mapfile.write_tile(layer, tile, &tile_data).unwrap();
            }
        }

        if self.gen_heights.refresh(&mut self.watcher) {
            self.mapfile.clear_generated(LayerType::Heights);
            self.mapfile.clear_generated(LayerType::Normals);
            self.tile_cache[LayerType::Heights.index()].clear_generated();
            self.tile_cache[LayerType::Normals.index()].clear_generated();
        }
        if self.gen_normals.refresh(&mut self.watcher) {
            self.mapfile.clear_generated(LayerType::Normals);
            self.tile_cache[LayerType::Normals.index()].clear_generated();
        }

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
                        attributes: self.shader.input_attributes(),
                    }],
                    sample_count: 1,
                    sample_mask: !0,
                    alpha_to_coverage_enabled: false,
                }),
            );
        }

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });

        let missing_heights = self.tile_cache[LayerType::Heights.index()].upload_tiles(
            device,
            &mut encoder,
            &self.gpu_state.heights,
            &mut self.mapfile,
        );
        let missing_normals = self.tile_cache[LayerType::Normals.index()].upload_tiles(
            device,
            &mut encoder,
            &self.gpu_state.normals,
            &mut self.mapfile,
        );
        self.tile_cache[LayerType::Colors.index()].upload_tiles(
            device,
            &mut encoder,
            &self.gpu_state.albedo,
            &mut self.mapfile,
        );

        let normals_resolution = self.tile_cache[LayerType::Normals.index()].resolution();
        let normals_border = self.tile_cache[LayerType::Normals.index()].border();
        let normals_row_pitch = self.tile_cache[LayerType::Normals.index()].row_pitch();
        for node in missing_normals.into_iter().take(1) {
            let spacing = node.side_length() / (normals_resolution - normals_border * 2) as f32;
            let position = node.bounds().min
                - cgmath::Vector3::new(spacing, 0.0, spacing) * normals_border as f32;
            self.gen_heights.run(
                device,
                &mut encoder,
                &self.gpu_state,
                (normals_resolution + 1, normals_resolution + 1, 1),
                &GenHeightsUniforms {
                    position: [position.x, position.z],
                    base_heights_step: 32.0,
                    step: spacing,
                },
            );
            self.gen_normals.run(
                device,
                &mut encoder,
                &self.gpu_state,
                ((normals_resolution + 7) / 8, (normals_resolution + 7) / 8, 1),
                &GenNormalsUniforms { position: [position.x, position.z], spacing },
            );

            let size = normals_resolution as u64 * normals_row_pitch as u64;
            let download = device.create_buffer(&wgpu::BufferDescriptor {
                size,
                usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::MAP_READ,
            });
            let tile = self.tile_cache[LayerType::Normals.index()].tile_for_node(node).unwrap();

            encoder.copy_texture_to_buffer(
                wgpu::TextureCopyView {
                    texture: &self.gpu_state.normals_staging,
                    mip_level: 0,
                    array_layer: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                },
                wgpu::BufferCopyView {
                    buffer: &download,
                    offset: 0,
                    row_pitch: normals_row_pitch as u32,
                    image_height: normals_resolution as u32,
                },
                wgpu::Extent3d {
                    width: normals_resolution as u32,
                    height: normals_resolution as u32,
                    depth: 1,
                },
            );

            self.pending_tiles.push((LayerType::Normals, tile, download));
        }
        let heights_resolution = self.tile_cache[LayerType::Heights.index()].resolution();
        let heights_row_pitch = self.tile_cache[LayerType::Heights.index()].row_pitch();
        for node in missing_heights.into_iter().take(32) {
            let step = node.side_length() / (heights_resolution - 1) as f32;
            let position = node.bounds().min;
            self.gen_heights.run(
                device,
                &mut encoder,
                &self.gpu_state,
                (heights_resolution, heights_resolution, 1),
                &GenHeightsUniforms {
                    position: [position.x, position.z],
                    base_heights_step: 32.0,
                    step,
                },
            );

            let size = heights_resolution as u64 * heights_row_pitch as u64;
            let download = device.create_buffer(&wgpu::BufferDescriptor {
                size,
                usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::MAP_READ,
            });
            let tile = self.tile_cache[LayerType::Heights.index()].tile_for_node(node).unwrap();

            encoder.copy_texture_to_buffer(
                wgpu::TextureCopyView {
                    texture: &self.gpu_state.heights_staging,
                    mip_level: 0,
                    array_layer: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                },
                wgpu::BufferCopyView {
                    buffer: &download,
                    offset: 0,
                    row_pitch: heights_row_pitch as u32,
                    image_height: heights_resolution as u32,
                },
                wgpu::Extent3d {
                    width: heights_resolution as u32,
                    height: heights_resolution as u32,
                    depth: 1,
                },
            );

            self.pending_tiles.push((LayerType::Heights, tile, download));
        }

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
