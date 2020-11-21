//! Terra is a large scale terrain generation and rendering library built on top of wgpu.
#![feature(async_closure)]
#![feature(is_sorted)]
#![feature(non_ascii_idents)]
#![feature(stmt_expr_attributes)]
#![feature(test)]
#![feature(with_options)]
#![allow(confusable_idents)]

#[cfg(test)]
extern crate test;

#[macro_use]
extern crate lazy_static;
extern crate rshader;

mod cache;
mod coordinates;
mod generate;
mod gpu_state;
mod mapfile;
mod sky;
mod srgb;
mod stream;
mod terrain;
mod utils;

use crate::terrain::quadtree::node::VNode;
use crate::terrain::quadtree::render::NodeState;
use crate::terrain::tile_cache::{LayerType, TileCache};
use anyhow::Error;
use gpu_state::GpuState;
use std::mem;
use std::sync::Arc;
use terrain::quadtree::QuadTree;

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

    gpu_state: GpuState,
    quadtree: QuadTree,
    mapfile: Arc<MapFile>,
    tile_cache: TileCache,
}
impl Terrain {
    pub fn new(
        device: &wgpu::Device,
        queue: &mut wgpu::Queue,
        mapfile: MapFile,
    ) -> Result<Self, Error> {
        let mapfile = Arc::new(mapfile);
        let tile_cache = TileCache::new(
            Arc::clone(&mapfile),
            crate::generate::generators(mapfile.layers(), device),
            512,
        );
        let quadtree = QuadTree::new(tile_cache.resolution(LayerType::Displacements) - 1);

        let shader = rshader::ShaderSet::simple(
            rshader::shader_source!("shaders", "version", "a.vert"),
            rshader::shader_source!("shaders", "version", "pbr", "a.frag", "atmosphere"),
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

        let (noise, sky, transmittance, inscattering);
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("load_textures".into()),
            });
            noise = mapfile.read_texture(device, &mut encoder, "noise")?;
            sky = mapfile.read_texture(device, &mut encoder, "sky")?;
            transmittance = mapfile.read_texture(device, &mut encoder, "transmittance")?;
            inscattering = mapfile.read_texture(device, &mut encoder, "inscattering")?;
            queue.submit(Some(encoder.finish()));
        }

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
            transmittance,
            inscattering,
            bc4_staging,
            bc5_staging,
            tile_cache: tile_cache.make_cache_textures(device),
        };

        let sky_shader = rshader::ShaderSet::simple(
            rshader::shader_source!("shaders", "version", "sky.vert"),
            rshader::shader_source!("shaders", "version", "pbr", "sky.frag", "atmosphere"),
        )
        .unwrap();
        let sky_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            size: std::mem::size_of::<SkyUniformBlock>() as u64,
            usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::UNIFORM,
            label: Some("sky.uniforms".into()),
            mapped_at_creation: false,
        });

        Ok(Self {
            bindgroup_pipeline: None,
            shader,

            uniform_buffer,
            vertex_buffer,
            index_buffer,
            index_buffer_partial,

            sky_shader,
            sky_uniform_buffer,
            sky_bindgroup_pipeline: None,

            gpu_state,
            quadtree,
            mapfile,
            tile_cache,
        })
    }

    pub fn loading(
        &mut self,
        device: &wgpu::Device,
        queue: &mut wgpu::Queue,
        camera: mint::Point3<f64>,
    ) -> bool {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        self.quadtree.update_cache(&mut self.tile_cache, camera);
        self.tile_cache.refresh_tile_generators();
        self.tile_cache.upload_tiles(device, &mut encoder, &self.gpu_state.tile_cache);
        self.tile_cache.generate_tiles(&self.mapfile, device, &mut encoder, &self.gpu_state);

        queue.submit(Some(encoder.finish()));

        VNode::roots().iter().copied().any(|root| {
            !self.tile_cache.contains(root, LayerType::Heightmaps)
                || !self.tile_cache.contains(root, LayerType::Albedo)
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
        self.quadtree.update_cache(&mut self.tile_cache, camera);
        if self.shader.refresh() {
            self.bindgroup_pipeline = None;
        }

        if self.bindgroup_pipeline.is_none() {
            let (bind_group, bind_group_layout) = self.gpu_state.bind_group_for_shader(
                device,
                &self.shader,
                Some(wgpu::BindingResource::Buffer {
                    buffer: &self.uniform_buffer,
                    offset: 0,
                    size: None,
                }),
            );
            let render_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                    label: None,
                });
            self.bindgroup_pipeline = Some((
                bind_group,
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    layout: Some(&render_pipeline_layout),
                    vertex_stage: wgpu::ProgrammableStageDescriptor {
                        module: &device.create_shader_module(wgpu::ShaderModuleSource::SpirV(
                            self.shader.vertex().into(),
                        )),
                        entry_point: "main",
                    },
                    fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                        module: &device.create_shader_module(wgpu::ShaderModuleSource::SpirV(
                            self.shader.fragment().into(),
                        )),
                        entry_point: "main",
                    }),
                    rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: wgpu::CullMode::Front,
                        ..Default::default()
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
                        stencil: Default::default(),
                    }),
                    vertex_state: wgpu::VertexStateDescriptor {
                        index_format: wgpu::IndexFormat::Uint16,
                        vertex_buffers: &[wgpu::VertexBufferDescriptor {
                            stride: std::mem::size_of::<NodeState>() as u64,
                            step_mode: wgpu::InputStepMode::Instance,
                            attributes: self.shader.input_attributes().into(),
                        }],
                    },
                    sample_count: 1,
                    sample_mask: !0,
                    alpha_to_coverage_enabled: false,
                    label: None,
                }),
            ));
        }

        if self.sky_shader.refresh() {
            self.sky_bindgroup_pipeline = None;
        }
        if self.sky_bindgroup_pipeline.is_none() {
            let (bind_group, bind_group_layout) = self.gpu_state.bind_group_for_shader(
                device,
                &self.sky_shader,
                Some(wgpu::BindingResource::Buffer {
                    buffer: &self.sky_uniform_buffer,
                    offset: 0,
                    size: None,
                }),
            );
            let render_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    bind_group_layouts: [&bind_group_layout][..].into(),
                    push_constant_ranges: &[],
                    label: None,
                });
            self.sky_bindgroup_pipeline = Some((
                bind_group,
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    layout: Some(&render_pipeline_layout),
                    vertex_stage: wgpu::ProgrammableStageDescriptor {
                        module: &device.create_shader_module(wgpu::ShaderModuleSource::SpirV(
                            self.sky_shader.vertex().into(),
                        )),
                        entry_point: "main",
                    },
                    fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                        module: &device.create_shader_module(wgpu::ShaderModuleSource::SpirV(
                            self.sky_shader.fragment().into(),
                        )),
                        entry_point: "main",
                    }),
                    rasterization_state: Some(wgpu::RasterizationStateDescriptor::default()),
                    primitive_topology: wgpu::PrimitiveTopology::TriangleList,
                    color_states: &[wgpu::ColorStateDescriptor {
                        format: wgpu::TextureFormat::Bgra8UnormSrgb,
                        color_blend: wgpu::BlendDescriptor::REPLACE,
                        alpha_blend: wgpu::BlendDescriptor::REPLACE,
                        write_mask: wgpu::ColorWrite::ALL,
                    }],
                    depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_write_enabled: false,
                        depth_compare: wgpu::CompareFunction::GreaterEqual,
                        stencil: Default::default(),
                    }),
                    vertex_state: wgpu::VertexStateDescriptor {
                        index_format: wgpu::IndexFormat::Uint16,
                        vertex_buffers: &[],
                    },
                    sample_count: 1,
                    sample_mask: !0,
                    alpha_to_coverage_enabled: false,
                    label: None,
                }),
            ));
        }

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        self.tile_cache.refresh_tile_generators();
        self.tile_cache.upload_tiles(device, &mut encoder, &self.gpu_state.tile_cache);
        self.tile_cache.generate_tiles(&self.mapfile, device, &mut encoder, &self.gpu_state);

        let mut do_render = true;
        for &root in &VNode::roots() {
            if !self.tile_cache.contains(root, LayerType::Heightmaps)
                || !self.tile_cache.contains(root, LayerType::Albedo)
                || !self.tile_cache.contains(root, LayerType::Roughness)
            {
                do_render = false;
                break;
            }
        }

        let camera_frustum = collision::Frustum::from_matrix4(view_proj.into());
        if do_render {
            self.quadtree.update_visibility(&self.tile_cache, camera, camera_frustum);
            self.quadtree.prepare_vertex_buffer(
                device,
                &mut encoder,
                &mut self.vertex_buffer,
                &self.tile_cache,
            );
        }

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

            rpass.push_debug_group("Terrain");
            rpass.set_pipeline(&self.bindgroup_pipeline.as_ref().unwrap().1);
            rpass.set_bind_group(0, &self.bindgroup_pipeline.as_ref().unwrap().0, &[]);

            if do_render {
                self.quadtree.render(
                    &mut rpass,
                    &self.vertex_buffer,
                    &self.index_buffer,
                    &self.index_buffer_partial,
                );
            }
            rpass.pop_debug_group();
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

            rpass.push_debug_group("Atmosphere");
            rpass.set_pipeline(&self.sky_bindgroup_pipeline.as_ref().unwrap().1);
            rpass.set_bind_group(0, &self.sky_bindgroup_pipeline.as_ref().unwrap().0, &[]);
            rpass.draw(0..3, 0..1);
            rpass.pop_debug_group();
        }

        queue.submit(Some(encoder.finish()));
    }
}
