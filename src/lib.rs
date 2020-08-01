//! Terra is a large scale terrain generation and rendering library built on top of wgpu.
#![feature(async_closure)]
#![feature(stmt_expr_attributes)]
#![feature(with_options)]
#![feature(is_sorted)]

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
use crate::mapfile::TileState;
use crate::terrain::quadtree::node::VNode;
use crate::terrain::quadtree::render::NodeState;
use crate::terrain::tile_cache::{LayerType, TileCache};
use anyhow::Error;
use cgmath::Vector2;
use futures::executor;
use gpu_state::GpuState;
use std::mem;
use terrain::quadtree::QuadTree;
use vec_map::VecMap;
// use wgpu_glyph::{GlyphBrush, Section};

pub use crate::mapfile::MapFile;
pub use generate::MapFileBuilder;

#[repr(C)]
#[derive(Copy, Clone)]
struct UniformBlock {
    view_proj: mint::ColumnMatrix4<f32>,
    camera: mint::Point3<f64>,
    padding: f64,
}
unsafe impl bytemuck::Pod for UniformBlock {}
unsafe impl bytemuck::Zeroable for UniformBlock {}

#[repr(C)]
#[derive(Copy, Clone)]
struct SkyUniformBlock {
    view_proj: mint::ColumnMatrix4<f32>,
    camera: mint::Point3<f64>,
    padding: f64,
}
unsafe impl bytemuck::Pod for SkyUniformBlock {}
unsafe impl bytemuck::Zeroable for SkyUniformBlock {}

pub struct Terrain {
    shader: rshader::ShaderSet,
    bindgroup_pipeline: Option<(wgpu::BindGroup, wgpu::RenderPipeline)>,
    uniform_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_buffer_partial: wgpu::Buffer,

    sky_shader: rshader::ShaderSet,
    sky_bindgroup_pipeline: Option<(wgpu::BindGroup, wgpu::RenderPipeline)>,
    sky_uniform_buffer: wgpu::Buffer,

    watcher: rshader::ShaderDirectoryWatcher,
    gen_heightmaps: ComputeShader<GenHeightmapsUniforms>,
    gen_displacements: ComputeShader<GenDisplacementsUniforms>,
    gen_normals: ComputeShader<GenNormalsUniforms>,
    gpu_state: GpuState,
    quadtree: QuadTree,
    mapfile: MapFile,
    tile_cache: TileCache,

    pending_tiles: Vec<(LayerType, VNode, wgpu::Buffer)>,
    // glyph_brush: GlyphBrush<'static, ()>,
}
impl Terrain {
    pub fn new(
        device: &wgpu::Device,
        queue: &mut wgpu::Queue,
        mut mapfile: MapFile,
    ) -> Result<Self, Error> {
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
            label: Some("terrain.uniforms".into()),
            mapped_at_creation: false,
        });
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            size: (std::mem::size_of::<NodeState>() * 1024) as u64,
            usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::VERTEX,
            label: Some("terrain.vertex_buffer".into()),
            mapped_at_creation: false,
        });
        let (index_buffer, index_buffer_partial) = {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("generate_index_buffers".into()),
            });
            let r = quadtree.create_index_buffers(device, &mut encoder);
            queue.submit(Some(encoder.finish()));
            r
        };

        let (noise, sky) = {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("load_textures".into()),
            });
            let noise = mapfile.read_texture(device, &mut encoder, "noise")?;
            let sky = mapfile.read_texture(device, &mut encoder, "sky")?;
            queue.submit(Some(encoder.finish()));
            (noise, sky)
        };
        let bc4_staging = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d { width: 256, height: 256, depth: 1 },
            format: wgpu::TextureFormat::Rg32Uint,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsage::COPY_SRC
                | wgpu::TextureUsage::COPY_DST
                | wgpu::TextureUsage::STORAGE
                | wgpu::TextureUsage::SAMPLED,
            label: None,
        });
        let bc5_staging = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d { width: 256, height: 256, depth: 1 },
            format: wgpu::TextureFormat::Rgba32Uint,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsage::COPY_SRC
                | wgpu::TextureUsage::COPY_DST
                | wgpu::TextureUsage::STORAGE
                | wgpu::TextureUsage::SAMPLED,
            label: None,
        });

        let gpu_state = GpuState {
            noise,
            sky,
            bc4_staging,
            bc5_staging,
            tile_cache: tile_cache.make_cache_textures(device),
        };

        let sky_shader = rshader::ShaderSet::simple(
            &mut watcher,
            rshader::shader_source!("shaders", "version", "sky.vert"),
            rshader::shader_source!("shaders", "version", "sky.frag"),
        )
        .unwrap();
        let sky_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            size: std::mem::size_of::<SkyUniformBlock>() as u64,
            usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::UNIFORM,
            label: Some("sky.uniforms".into()),
            mapped_at_creation: false,
        });

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
                rshader::shader_source!("shaders", "version", "hash", "gen-normals.comp"),
            )
            .unwrap(),
        );

        // TODO: only clear if shader has changed?
        mapfile.clear_generated(LayerType::Displacements).unwrap();
        mapfile.clear_generated(LayerType::Normals).unwrap();
        mapfile.clear_generated(LayerType::Heightmaps).unwrap();
        mapfile.clear_generated(LayerType::Albedo).unwrap();

        // let font: &'static [u8] = include_bytes!("../assets/UbuntuMono/UbuntuMono-R.ttf");
        // let glyph_brush = wgpu_glyph::GlyphBrushBuilder::using_font_bytes(font)
        //     .unwrap()
        //     .build(device, wgpu::TextureFormat::Bgra8UnormSrgb);

        Ok(Self {
            bindgroup_pipeline: None,
            watcher,
            shader,

            uniform_buffer,
            vertex_buffer,
            index_buffer,
            index_buffer_partial,

            sky_shader,
            sky_uniform_buffer,
            sky_bindgroup_pipeline: None,

            gen_heightmaps,
            gen_displacements,
            gen_normals,

            gpu_state,
            quadtree,
            mapfile,
            tile_cache,
            // glyph_brush,
            pending_tiles: Vec::new(),
        })
    }

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &mut wgpu::Queue,
        color_buffer: &wgpu::TextureView,
        depth_buffer: &wgpu::TextureView,
        _frame_size: (u32, u32),
        view_proj: mint::ColumnMatrix4<f32>,
        camera: mint::Point3<f64>,
    ) {
        for (layer, node, buffer) in self.pending_tiles.drain(..) {
            let resolution_blocks = self.tile_cache.resolution_blocks(layer) as usize;
            let bytes_per_block = self.tile_cache.bytes_per_block(layer);
            let row_pitch = self.tile_cache.row_pitch(layer);

            let row_bytes = resolution_blocks * bytes_per_block;
            let size = row_pitch * resolution_blocks;

            let buffer_slice = buffer.slice(..size as u64);
            let future = buffer_slice.map_async(wgpu::MapMode::Read);

            device.poll(wgpu::Maintain::Wait);

            if executor::block_on(future).is_ok() {
                let mut tile_data =
                    vec![0; resolution_blocks * resolution_blocks * bytes_per_block];
                let buffer_data = &*buffer_slice.get_mapped_range();
                for row in 0..resolution_blocks {
                    tile_data[row * row_bytes..][..row_bytes]
                        .copy_from_slice(&buffer_data[row * row_pitch..][..row_bytes]);
                }
                self.mapfile.write_tile(layer, node, &tile_data, false).unwrap();
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
            self.tile_cache.clear_generated(LayerType::Albedo);
        }

        self.quadtree.update_cache(&mut self.tile_cache, camera);
        if self.shader.refresh(&mut self.watcher) {
            self.bindgroup_pipeline = None;
        }

        if self.bindgroup_pipeline.is_none() {
            let (bind_group, bind_group_layout) = self.gpu_state.bind_group_for_shader(
                device,
                &self.shader,
                Some(self.uniform_buffer.slice(0..mem::size_of::<UniformBlock>() as u64)),
            );
            let render_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    bind_group_layouts: vec![&bind_group_layout].into(),
                    push_constant_ranges: vec![].into(),
                });
            self.bindgroup_pipeline = Some((
                bind_group,
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    layout: &render_pipeline_layout,
                    vertex_stage: wgpu::ProgrammableStageDescriptor {
                        module: &device.create_shader_module(wgpu::ShaderModuleSource::SpirV(
                            self.shader.vertex().into(),
                        )),
                        entry_point: "main".into(),
                    },
                    fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                        module: &device.create_shader_module(wgpu::ShaderModuleSource::SpirV(
                            self.shader.fragment().into(),
                        )),
                        entry_point: "main".into(),
                    }),
                    rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: wgpu::CullMode::Front,
                        ..Default::default()
                    }),
                    primitive_topology: wgpu::PrimitiveTopology::TriangleList,
                    color_states: [wgpu::ColorStateDescriptor {
                        format: wgpu::TextureFormat::Bgra8UnormSrgb,
                        color_blend: wgpu::BlendDescriptor::REPLACE,
                        alpha_blend: wgpu::BlendDescriptor::REPLACE,
                        write_mask: wgpu::ColorWrite::ALL,
                    }][..]
                        .into(),
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
                        vertex_buffers: [wgpu::VertexBufferDescriptor {
                            stride: std::mem::size_of::<NodeState>() as u64,
                            step_mode: wgpu::InputStepMode::Instance,
                            attributes: self.shader.input_attributes().into(),
                        }][..]
                            .into(),
                    },
                    sample_count: 1,
                    sample_mask: !0,
                    alpha_to_coverage_enabled: false,
                }),
            ));
        }

        if self.sky_shader.refresh(&mut self.watcher) {
            self.sky_bindgroup_pipeline = None;
        }
        if self.sky_bindgroup_pipeline.is_none() {
            let (bind_group, bind_group_layout) = self.gpu_state.bind_group_for_shader(
                device,
                &self.sky_shader,
                Some(self.sky_uniform_buffer.slice(0..mem::size_of::<SkyUniformBlock>() as u64)),
            );
            let render_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    bind_group_layouts: [&bind_group_layout][..].into(),
                    push_constant_ranges: vec![].into(),
                });
            self.sky_bindgroup_pipeline = Some((
                bind_group,
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    layout: &render_pipeline_layout,
                    vertex_stage: wgpu::ProgrammableStageDescriptor {
                        module: &device.create_shader_module(wgpu::ShaderModuleSource::SpirV(
                            self.sky_shader.vertex().into(),
                        )),
                        entry_point: "main".into(),
                    },
                    fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                        module: &device.create_shader_module(wgpu::ShaderModuleSource::SpirV(
                            self.sky_shader.fragment().into(),
                        )),
                        entry_point: "main".into(),
                    }),
                    rasterization_state: Some(wgpu::RasterizationStateDescriptor::default()),
                    primitive_topology: wgpu::PrimitiveTopology::TriangleList,
                    color_states: [wgpu::ColorStateDescriptor {
                        format: wgpu::TextureFormat::Bgra8UnormSrgb,
                        color_blend: wgpu::BlendDescriptor::REPLACE,
                        alpha_blend: wgpu::BlendDescriptor::REPLACE,
                        write_mask: wgpu::ColorWrite::ALL,
                    }][..]
                        .into(),
                    depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_write_enabled: false,
                        depth_compare: wgpu::CompareFunction::GreaterEqual,
                        stencil_front: Default::default(),
                        stencil_back: Default::default(),
                        stencil_read_mask: 0,
                        stencil_write_mask: 0,
                    }),
                    vertex_state: wgpu::VertexStateDescriptor {
                        index_format: wgpu::IndexFormat::Uint16,
                        vertex_buffers: vec![].into(),
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
            let mut layer_missing = self.tile_cache.upload_tiles(
                device,
                &mut encoder,
                &texture,
                &mut self.mapfile,
                LayerType::from_index(i),
            );
            layer_missing.sort_by_key(|n| n.level());
            missing.insert(i, layer_missing);
        }

        let heightmaps_resolution = self.tile_cache.resolution(LayerType::Heightmaps);
        let heightmaps_border = self.tile_cache.border(LayerType::Heightmaps);

        let normals_resolution = self.tile_cache.resolution(LayerType::Normals);
        let normals_border = self.tile_cache.border(LayerType::Normals);
        let normals_row_pitch = self.tile_cache.row_pitch(LayerType::Normals);

        for (i, node) in missing.remove(LayerType::Normals.index()).unwrap().into_iter().enumerate()
        {
            if node.level() > 0 && i >= 16 {
                continue;
            }

            let heightmaps_slot = match self.tile_cache.get_slot(node) {
                Some(slot) => slot as i32,
                None => continue,
            };

            let parent_slot = node
                .parent()
                .map(|(p, idx)| (self.tile_cache.get_slot(p).unwrap() as i32, idx))
                .unwrap_or((-1, 0));

            let spacing =
                node.aprox_side_length() / (normals_resolution - normals_border * 2) as f32;
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
                    // let position = node.bounds().min
                    //     - cgmath::Vector3::new(spacing, 0.0, spacing) * heightmaps_border as f32;
                    self.gen_heightmaps.run(
                        device,
                        &mut encoder,
                        &self.gpu_state,
                        ((heightmaps_resolution + 7) / 8, (heightmaps_resolution + 7) / 8, 1),
                        &GenHeightmapsUniforms {
                            position: [0.0, 0.0],
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

            let albedo_slot =
                if self.tile_cache.slot_valid(normals_slot as usize, LayerType::Albedo) {
                    -1
                } else {
                    normals_slot
                };

            let cspace_origin =
                node.cell_position_cspace(0, 0, normals_border as u16, normals_resolution as u16);
            let cspace_origin_dx =
                node.cell_position_cspace(1, 0, normals_border as u16, normals_resolution as u16);
            let cspace_origin_dy =
                node.cell_position_cspace(0, 1, normals_border as u16, normals_resolution as u16);

            self.gen_normals.run(
                device,
                &mut encoder,
                &self.gpu_state,
                ((normals_resolution + 3) / 4, (normals_resolution + 3) / 4, 1),
                &GenNormalsUniforms {
                    heightmaps_origin: [
                        (heightmaps_border - normals_border) as i32,
                        (heightmaps_border - normals_border) as i32,
                    ],
                    cspace_origin: [cspace_origin.x, cspace_origin.y, cspace_origin.z, 0.0],
                    cspace_dx: [
                        cspace_origin_dx.x - cspace_origin.x,
                        cspace_origin_dx.y - cspace_origin.y,
                        cspace_origin_dx.z - cspace_origin.z,
                        0.0,
                    ],
                    cspace_dy: [
                        cspace_origin_dy.x - cspace_origin.x,
                        cspace_origin_dy.y - cspace_origin.y,
                        cspace_origin_dy.z - cspace_origin.z,
                        0.0,
                    ],
                    spacing,
                    heightmaps_slot,
                    normals_slot,
                    albedo_slot,
                    parent_slot: parent_slot.0,
                    parent_origin: [
                        if parent_slot.1 % 2 == 0 {
                            normals_border / 2
                        } else {
                            (normals_resolution - normals_border) / 2
                        },
                        if parent_slot.1 / 2 == 0 {
                            normals_border / 2
                        } else {
                            (normals_resolution - normals_border) / 2
                        },
                    ],
                    padding: 0,
                },
            );
            self.tile_cache.set_slot_valid(normals_slot as usize, LayerType::Normals);
            if albedo_slot >= 0 {
                self.tile_cache.set_slot_valid(albedo_slot as usize, LayerType::Albedo);
            }

            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                size: 1024 * 1024 * 4, // TODO
                usage: wgpu::BufferUsage::COPY_SRC | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
                label: None,
            });
            encoder.copy_texture_to_buffer(
                wgpu::TextureCopyView {
                    texture: &self.gpu_state.bc5_staging,
                    mip_level: 0,
                    origin: wgpu::Origin3d::default(),
                },
                wgpu::BufferCopyView {
                    buffer: &buffer,
                    layout: wgpu::TextureDataLayout {
                        bytes_per_row: 4096,
                        rows_per_image: normals_resolution as u32 / 4,
                        offset: 0,
                    },
                },
                wgpu::Extent3d {
                    width: normals_resolution as u32 / 4,
                    height: normals_resolution as u32 / 4,
                    depth: 1,
                },
            );
            encoder.copy_buffer_to_texture(
                wgpu::BufferCopyView {
                    buffer: &buffer,
                    layout: wgpu::TextureDataLayout {
                        bytes_per_row: 4096,
                        rows_per_image: normals_resolution as u32,
                        offset: 0,
                    },
                },
                wgpu::TextureCopyView {
                    texture: &self.gpu_state.tile_cache[LayerType::Normals],
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: normals_slot as u32 },
                },
                wgpu::Extent3d {
                    width: normals_resolution as u32,
                    height: normals_resolution as u32,
                    depth: 1,
                },
            );

            if let TileState::Missing = self.mapfile.tile_state(LayerType::Normals, node).unwrap() {
                let size = normals_resolution as u64 * normals_row_pitch as u64;
                let download = device.create_buffer(&wgpu::BufferDescriptor {
                    size,
                    usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::MAP_READ,
                    label: Some("download_tiles.normals".into()),
                    mapped_at_creation: false,
                });
                encoder.copy_texture_to_buffer(
                    wgpu::TextureCopyView {
                        texture: &self.gpu_state.tile_cache[LayerType::Normals],
                        mip_level: 0,
                        origin: wgpu::Origin3d { x: 0, y: 0, z: normals_slot as u32 },
                    },
                    wgpu::BufferCopyView {
                        buffer: &download,
                        layout: wgpu::TextureDataLayout {
                            offset: 0,
                            bytes_per_row: normals_row_pitch as u32,
                            rows_per_image: 0,
                        },
                    },
                    wgpu::Extent3d {
                        width: normals_resolution as u32,
                        height: normals_resolution as u32,
                        depth: 1,
                    },
                );
                self.pending_tiles.push((LayerType::Normals, node, download));
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

            if let TileState::Missing =
                self.mapfile.tile_state(LayerType::Displacements, node).unwrap()
            {
                let size = displacements_resolution as u64 * displacements_row_pitch as u64;
                let download = device.create_buffer(&wgpu::BufferDescriptor {
                    size,
                    usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::MAP_READ,
                    label: Some("download_tiles.displacements".into()),
                    mapped_at_creation: false,
                });
                encoder.copy_texture_to_buffer(
                    wgpu::TextureCopyView {
                        texture: &self.gpu_state.tile_cache[LayerType::Displacements],
                        mip_level: 0,
                        origin: wgpu::Origin3d { x: 0, y: 0, z: displacements_slot as u32 },
                    },
                    wgpu::BufferCopyView {
                        buffer: &download,
                        layout: wgpu::TextureDataLayout {
                            offset: 0,
                            bytes_per_row: displacements_row_pitch as u32,
                            rows_per_image: 0,
                        },
                    },
                    wgpu::Extent3d {
                        width: displacements_resolution as u32,
                        height: displacements_resolution as u32,
                        depth: 1,
                    },
                );
                self.pending_tiles.push((LayerType::Displacements, node, download));
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

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            size: mem::size_of::<UniformBlock>() as u64,
            usage: wgpu::BufferUsage::MAP_WRITE | wgpu::BufferUsage::COPY_SRC,
            label: Some("terrain_uniforms_upload".into()),
            mapped_at_creation: true,
        });
        let mut buffer_view = buffer.slice(..).get_mapped_range_mut();
        bytemuck::cast_slice_mut(&mut *buffer_view)[0] =
            UniformBlock { view_proj, camera, padding: 0.0 };

        drop(buffer_view);
        buffer.unmap();
        encoder.copy_buffer_to_buffer(
            &buffer,
            0,
            &self.uniform_buffer,
            0,
            mem::size_of::<UniformBlock>() as u64,
        );
        encoder.copy_buffer_to_buffer(
            &buffer,
            0,
            &self.sky_uniform_buffer,
            0,
            mem::size_of::<SkyUniformBlock>() as u64,
        );

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: [wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: color_buffer,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.3, b: 0.8, a: 1.0 }),
                        store: true,
                    },
                }][..]
                    .into(),
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                    attachment: depth_buffer,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: true,
                    }),
                    stencil_ops: None,
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
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: [wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: color_buffer,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: true },
                }][..]
                    .into(),
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                    attachment: depth_buffer,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: true }),
                    stencil_ops: None,
                }),
            });

            rpass.set_pipeline(&self.sky_bindgroup_pipeline.as_ref().unwrap().1);
            rpass.set_bind_group(0, &self.sky_bindgroup_pipeline.as_ref().unwrap().0, &[]);
            rpass.draw(0..3, 0..1);
        }

        // {
        //     use std::fmt::Write;
        //     let mut text = String::new();
        //     writeln!(
        //         &mut text,
        //         "tile_cache: {} / {}",
        //         self.tile_cache.utilization(),
        //         self.tile_cache.capacity()
        //     )
        //     .unwrap();

        //     writeln!(&mut text, "x = {}\ny = {}\nz = {}", camera.x, camera.y, camera.z).unwrap();

        //     self.glyph_brush.queue(Section {
        //         text: &text,
        //         screen_position: (10.0, 10.0),
        //         ..Default::default()
        //     });
        // }

        // self.glyph_brush
        //     .draw_queued(device, &mut encoder, &frame.view, frame_size.0, frame_size.1)
        //     .unwrap();
        queue.submit(Some(encoder.finish()));
    }
}
