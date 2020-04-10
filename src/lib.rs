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

use crate::generate::{
    ComputeShader, GenDisplacementsUniforms, GenHeightmapsUniforms, GenNormalsUniforms,
};
use crate::terrain::quadtree::render::NodeState;
use crate::terrain::tile_cache::{LayerType, TileCache};
use cgmath::Vector2;
use futures::executor;
use gpu_state::GpuState;
use std::mem;
use terrain::quadtree::QuadTree;
use vec_map::VecMap;
use wgpu_glyph::{GlyphBrush, Section};

pub use crate::mapfile::MapFile;
pub use generate::{MapFileBuilder, TextureQuality, VertexQuality};

#[repr(C)]
#[derive(Copy, Clone)]
struct UniformBlock {
    local_origin: [f64; 3],
    padding2: f64,
    view_proj: mint::ColumnMatrix4<f32>,
    camera: mint::Point3<f32>,
    padding: f32,
}
unsafe impl bytemuck::Pod for UniformBlock {}
unsafe impl bytemuck::Zeroable for UniformBlock {}

pub struct Terrain {
    bindgroup_pipeline: Option<(wgpu::BindGroup, wgpu::RenderPipeline)>,

    watcher: rshader::ShaderDirectoryWatcher,
    shader: rshader::ShaderSet,

    uniform_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_buffer_partial: wgpu::Buffer,

    gen_heightmaps: ComputeShader<GenHeightmapsUniforms>,
    gen_displacements: ComputeShader<GenDisplacementsUniforms>,
    gen_normals: ComputeShader<GenNormalsUniforms>,
    gpu_state: GpuState,
    quadtree: QuadTree,
    mapfile: MapFile,
    tile_cache: TileCache,

    pending_tiles: Vec<(LayerType, usize, wgpu::Buffer)>,

    glyph_brush: GlyphBrush<'static, ()>,
}
impl Terrain {
    pub fn new(device: &wgpu::Device, queue: &mut wgpu::Queue, mut mapfile: MapFile) -> Self {
        let tile_cache = TileCache::new(mapfile.layers().clone(), 512);
        let quadtree = QuadTree::new(tile_cache.resolution(LayerType::Displacements) - 1);

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
            label: Some("terrain.uniforms"),
        });
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            size: (std::mem::size_of::<NodeState>() * 1024) as u64,
            usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::VERTEX,
            label: Some("terrain.vertex_buffer"),
        });
        let (index_buffer, index_buffer_partial) = quadtree.create_index_buffers(device);

        let noise = {
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("generate_noise") });
            let noise = mapfile.noise_texture(device, &mut encoder);
            queue.submit(&[encoder.finish()]);
            noise
        };

        let gpu_state = GpuState {
            noise,
            tile_cache: tile_cache.make_cache_textures(device),
        };

        let gen_heightmaps = ComputeShader::new(
            device,
            rshader::ShaderSet::compute_only(
                &mut watcher,
                rshader::shader_source!("shaders", "version", "hash", "gen-heightmaps.comp"),
            )
            .unwrap(),
        );
        let gen_displacements = ComputeShader::new(
            device,
            rshader::ShaderSet::compute_only(
                &mut watcher,
                rshader::shader_source!("shaders", "version", "gen-displacements.comp"),
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
        mapfile.clear_generated(LayerType::Displacements).unwrap();
        mapfile.clear_generated(LayerType::Normals).unwrap();
        mapfile.clear_generated(LayerType::Heightmaps).unwrap();
        mapfile.clear_generated(LayerType::Albedo).unwrap();

        let font: &'static [u8] = include_bytes!("../assets/UbuntuMono/UbuntuMono-R.ttf");
        let glyph_brush = wgpu_glyph::GlyphBrushBuilder::using_font_bytes(font).unwrap()
            .build(device, wgpu::TextureFormat::Bgra8UnormSrgb);

        Self {
            bindgroup_pipeline: None,
            watcher,
            shader,

            uniform_buffer,
            vertex_buffer,
            index_buffer,
            index_buffer_partial,

            gen_heightmaps,
            gen_displacements,
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
            let resolution = self.tile_cache.resolution(layer) as usize;
            let bytes_per_texel = self.tile_cache.bytes_per_texel(layer);
            let row_pitch = self.tile_cache.row_pitch(layer);
            let row_bytes = resolution * bytes_per_texel;
            let size = row_pitch * resolution;

            let future = buffer.map_read(0, size as u64);

            device.poll(wgpu::Maintain::Wait);

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

        if self.gen_heightmaps.refresh(&mut self.watcher) {
            self.mapfile.clear_generated(LayerType::Heightmaps).unwrap();
            self.mapfile.clear_generated(LayerType::Displacements).unwrap();
            self.mapfile.clear_generated(LayerType::Normals).unwrap();
            self.tile_cache.clear_generated(LayerType::Heightmaps);
            self.tile_cache.clear_generated(LayerType::Displacements);
            self.tile_cache.clear_generated(LayerType::Normals);
        }
        if self.gen_displacements.refresh(&mut self.watcher) {
            self.mapfile.clear_generated(LayerType::Displacements).unwrap();
            self.tile_cache.clear_generated(LayerType::Displacements);
        }
        if self.gen_normals.refresh(&mut self.watcher) {
            self.mapfile.clear_generated(LayerType::Normals).unwrap();
            self.tile_cache.clear_generated(LayerType::Normals);
            self.tile_cache.clear_generated(LayerType::Normals);
        }

        self.quadtree.update_cache(&mut self.tile_cache, camera);
        if self.shader.refresh(&mut self.watcher) {
            self.bindgroup_pipeline = None;
        }

        if self.bindgroup_pipeline.is_none() {
            let (bind_group, bind_group_layout) = self.gpu_state.bind_group_for_shader(
                device,
                &self.shader,
                Some(&wgpu::BindingResource::Buffer {
                    buffer: &self.uniform_buffer,
                    range: 0..mem::size_of::<UniformBlock>() as u64,
                }),
            );
            let render_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    bind_group_layouts: &[&bind_group_layout],
                });
            self.bindgroup_pipeline = Some((
                bind_group,
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    layout: &render_pipeline_layout,
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
                    vertex_state: wgpu::VertexStateDescriptor {
                        index_format: wgpu::IndexFormat::Uint16,
                        vertex_buffers: &[wgpu::VertexBufferDescriptor {
                            stride: std::mem::size_of::<NodeState>() as u64,
                            step_mode: wgpu::InputStepMode::Instance,
                            attributes: self.shader.input_attributes(),
                        }],
                    },
                    sample_count: 1,
                    sample_mask: !0,
                    alpha_to_coverage_enabled: false,
                }),
            ));
        }

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let mut missing = VecMap::new();
        for (i, texture) in &self.gpu_state.tile_cache {
            missing.insert(
                i,
                self.tile_cache.upload_tiles(
                    device,
                    &mut encoder,
                    &texture,
                    &mut self.mapfile,
                    LayerType::from_index(i),
                ),
            );
        }

        let heightmaps_resolution = self.tile_cache.resolution(LayerType::Heightmaps);
        let heightmaps_border = self.tile_cache.border(LayerType::Heightmaps);

        let normals_resolution = self.tile_cache.resolution(LayerType::Normals);
        let normals_border = self.tile_cache.border(LayerType::Normals);
        let normals_row_pitch = self.tile_cache.row_pitch(LayerType::Normals);
        for node in missing.remove(LayerType::Normals.index()).unwrap().into_iter().take(8) {
            let heightmaps_slot = match self.tile_cache.get_slot(node) {
                Some(slot) => slot as i32,
                None => continue,
            };

            let spacing = node.aprox_side_length() / (normals_resolution - normals_border * 2) as f32;
            // let position = node.bounds().min
            //     - cgmath::Vector3::new(spacing, 0.0, spacing) * normals_border as f32;

            let normals_slot = self.tile_cache.get_slot(node).unwrap() as i32;

            if !self.tile_cache.slot_valid(heightmaps_slot as usize, LayerType::Heightmaps) {
                let mut nodes_needed = vec![node];
                let mut parent = node.parent().unwrap().0;
                while !self.tile_cache.contains(parent, LayerType::Heightmaps) {
                    nodes_needed.push(parent);
                    parent = parent.parent().unwrap().0;
                }

                for node in nodes_needed.drain(..).rev() {
                    let (parent, parent_index) = node.parent().expect("root node missing");
                    let parent_offset = terrain::quadtree::node::OFFSETS[parent_index as usize];
                    let origin = [
                        heightmaps_border as i32 / 2,
                        heightmaps_resolution as i32 / 2 - heightmaps_border as i32 / 2,
                    ];
                    let in_slot = self.tile_cache.get_slot(parent).unwrap() as i32;
                    let out_slot = self.tile_cache.get_slot(node).unwrap() as i32;
                    let spacing = node.aprox_side_length()
                        / (heightmaps_resolution - heightmaps_border * 2 - 1) as f32;
                    let position = node.bounds().min
                        - cgmath::Vector3::new(spacing, 0.0, spacing) * heightmaps_border as f32;
                    self.gen_heightmaps.run(
                        device,
                        &mut encoder,
                        &self.gpu_state,
                        ((heightmaps_resolution + 7) / 8, (heightmaps_resolution + 7) / 8, 1),
                        &GenHeightmapsUniforms {
                            position: [position.x, position.z],
                            origin: [
                                origin[parent_offset.x as usize],
                                origin[parent_offset.y as usize],
                            ],
                            spacing,
                            in_slot,
                            out_slot,
                        },
                    );
                    self.tile_cache.set_slot_valid(out_slot as usize, LayerType::Heightmaps);
                }
            }

            self.gen_normals.run(
                device,
                &mut encoder,
                &self.gpu_state,
                ((normals_resolution + 7) / 8, (normals_resolution + 7) / 8, 1),
                &GenNormalsUniforms {
                    origin: [
                        (heightmaps_border - normals_border) as i32,
                        (heightmaps_border - normals_border) as i32,
                    ],
                    world_origin: [
                        node.bounds().min.x - (normals_border as f32 - 0.5) * spacing,
                        node.bounds().min.z - (normals_border as f32 - 0.5) * spacing,
                    ],
                    spacing,
                    heightmaps_slot,
                    normals_slot,
                },
            );
            self.tile_cache.set_slot_valid(normals_slot as usize, LayerType::Normals);
            self.tile_cache.set_slot_valid(normals_slot as usize, LayerType::Albedo);

            if let Some(tile) = self.tile_cache.tile_for_node(node, LayerType::Normals) {
                let size = normals_resolution as u64 * normals_row_pitch as u64;
                let download = device.create_buffer(&wgpu::BufferDescriptor {
                    size,
                    usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::MAP_READ,
                    label: Some("download_tiles.normals"),
                });
                encoder.copy_texture_to_buffer(
                    wgpu::TextureCopyView {
                        texture: &self.gpu_state.tile_cache[LayerType::Normals],
                        mip_level: 0,
                        array_layer: normals_slot as u32,
                        origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                    },
                    wgpu::BufferCopyView {
                        buffer: &download,
                        offset: 0,
                        bytes_per_row: normals_row_pitch as u32,
                        rows_per_image: normals_resolution as u32,
                    },
                    wgpu::Extent3d {
                        width: normals_resolution as u32,
                        height: normals_resolution as u32,
                        depth: 1,
                    },
                );
                self.pending_tiles.push((LayerType::Normals, tile, download));
            }
        }
        let displacements_resolution = self.tile_cache.resolution(LayerType::Displacements);
        let displacements_row_pitch = self.tile_cache.row_pitch(LayerType::Displacements);
        assert_eq!(self.tile_cache.border(LayerType::Displacements), 0);
        for node in missing.remove(LayerType::Displacements.index()).unwrap().into_iter() {
            let displacements_slot = self.tile_cache.get_slot(node).unwrap();

            let mut ancestor = node;
            let mut offset = Vector2::new(0, 0);
            let mut stride = (heightmaps_resolution - heightmaps_border * 2 - 1)
                / (displacements_resolution - 1);
            let mut generations = 0;
            while stride > 1 && ancestor.level() > 0 {
                offset += Vector2::new(ancestor.x() & 1, ancestor.y() & 1) * (1 << generations);
                ancestor = ancestor.parent().unwrap().0;
                generations += 1;
                stride /= 2;
            }

            let heightmaps_slot = self.tile_cache.get_slot(ancestor).unwrap();
            if !self.tile_cache.slot_valid(heightmaps_slot, LayerType::Heightmaps) {
                continue;
            }

            self.gen_displacements.run(
                device,
                &mut encoder,
                &self.gpu_state,
                ((displacements_resolution + 7) / 8, (displacements_resolution + 7) / 8, 1),
                &GenDisplacementsUniforms {
                    origin: [
                        (heightmaps_border
                            + (heightmaps_resolution - heightmaps_border * 2 - 1) * offset.x
                                / (1 << generations)) as i32,
                        (heightmaps_border
                            + (heightmaps_resolution - heightmaps_border * 2 - 1) * offset.y
                                / (1 << generations)) as i32,
                    ],
                    stride: stride as i32,
                    displacements_slot: displacements_slot as i32,
                    heightmaps_slot: heightmaps_slot as i32,
                },
            );
            self.tile_cache.set_slot_valid(displacements_slot as usize, LayerType::Displacements);

            if let Some(tile) = self.tile_cache.tile_for_node(node, LayerType::Displacements) {
                let size = displacements_resolution as u64 * displacements_row_pitch as u64;
                let download = device.create_buffer(&wgpu::BufferDescriptor {
                    size,
                    usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::MAP_READ,
                    label: Some("download_tiles.displacements"),
                });
                encoder.copy_texture_to_buffer(
                    wgpu::TextureCopyView {
                        texture: &self.gpu_state.tile_cache[LayerType::Displacements],
                        mip_level: 0,
                        array_layer: displacements_slot as u32,
                        origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                    },
                    wgpu::BufferCopyView {
                        buffer: &download,
                        offset: 0,
                        bytes_per_row: displacements_row_pitch as u32,
                        rows_per_image: displacements_resolution as u32,
                    },
                    wgpu::Extent3d {
                        width: displacements_resolution as u32,
                        height: displacements_resolution as u32,
                        depth: 1,
                    },
                );
                self.pending_tiles.push((LayerType::Displacements, tile, download));
            }
        }

        let camera_frustum = collision::Frustum::from_matrix4(view_proj.into());
        self.quadtree.update_visibility(&self.tile_cache, camera, camera_frustum);
        self.quadtree.prepare_vertex_buffer(
            device,
            &mut encoder,
            &mut self.vertex_buffer,
            &self.tile_cache,
        );

        let mapped = device.create_buffer_mapped(&wgpu::BufferDescriptor {
            size: mem::size_of::<UniformBlock>() as u64,
            usage: wgpu::BufferUsage::MAP_WRITE | wgpu::BufferUsage::COPY_SRC,
            label: Some("terrain_uniforms_upload"),
        });
        bytemuck::cast_slice_mut(mapped.data)[0] = UniformBlock {
            view_proj,
            camera,
            padding: 0.0,
            local_origin: [0.0, 6371000.0, 0.0],
            padding2: 0.0,
        };
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
            rpass.set_pipeline(&self.bindgroup_pipeline.as_ref().unwrap().1);
            rpass.set_bind_group(0, &self.bindgroup_pipeline.as_ref().unwrap().0, &[]);

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
            writeln!(
                &mut text,
                "tile_cache: {} / {}",
                self.tile_cache.utilization(),
                self.tile_cache.capacity()
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
