//! Terra is a large scale terrain generation and rendering library built on top of wgpu.
#![cfg_attr(test, feature(test))]

#[cfg(test)]
extern crate test;

#[macro_use]
extern crate lazy_static;
extern crate rshader;

mod asset;
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

use crate::cache::{LayerType, MeshCacheDesc, MeshType};
use crate::generate::MapFileBuilder;
use crate::mapfile::MapFile;
use crate::terrain::quadtree::node::VNode;
use crate::terrain::quadtree::render::NodeState;
use anyhow::Error;
use cache::{SingularLayerDesc, SingularLayerType, TextureFormat, UnifiedPriorityCache};
use cgmath::SquareMatrix;
use generate::ComputeShader;
use gpu_state::GpuState;
use maplit::hashmap;
use std::mem;
use std::sync::Arc;
use std::{collections::HashMap, convert::TryInto};
use terrain::quadtree::QuadTree;

pub use crate::generate::BLUE_MARBLE_URLS;

#[repr(C)]
#[derive(Copy, Clone)]
struct GlobalUniformBlock {
    view_proj: mint::ColumnMatrix4<f32>,
    view_proj_inverse: mint::ColumnMatrix4<f32>,
    camera: mint::Point3<f32>,
    sun_direction: mint::Vector3<f32>,
    padding: [f32; 2],
}
unsafe impl bytemuck::Pod for GlobalUniformBlock {}
unsafe impl bytemuck::Zeroable for GlobalUniformBlock {}

pub struct Terrain {
    shader: rshader::ShaderSet,
    bindgroup_pipeline: Option<(wgpu::BindGroup, wgpu::RenderPipeline)>,
    index_buffer: wgpu::Buffer,

    sky_shader: rshader::ShaderSet,
    sky_bindgroup_pipeline: Option<(wgpu::BindGroup, wgpu::RenderPipeline)>,
    aerial_perspective: ComputeShader<u32>,

    gpu_state: GpuState,
    quadtree: QuadTree,
    mapfile: Arc<MapFile>,

    cache: UnifiedPriorityCache,
}
impl Terrain {
    /// Create a new Terrain object.
    pub fn new(device: &wgpu::Device, queue: &mut wgpu::Queue) -> Result<Self, Error> {
        let mapfile = Arc::new(futures::executor::block_on(MapFileBuilder::new().build())?);
        let cache = UnifiedPriorityCache::new(
            device,
            Arc::clone(&mapfile),
            512,
            crate::generate::generators(
                mapfile.layers(),
                !device.features().contains(wgpu::Features::SHADER_FLOAT64),
            ),
            vec![MeshCacheDesc {
                size: 32,
                ty: MeshType::Grass,
                max_bytes_per_entry: 128 * 128 * 32,
                dimensions: 128 / 8,
                dependency_mask: LayerType::Displacements.bit_mask()
                    | LayerType::Albedo.bit_mask()
                    | LayerType::Normals.bit_mask(),
                level: VNode::LEVEL_CELL_2CM,
                generate: ComputeShader::new(
                    rshader::shader_source!(
                        "shaders",
                        "declarations.glsl",
                        "hash.glsl",
                        "gen-grass.comp"
                    ),
                    "gen-grass".to_string(),
                ),
                render: rshader::ShaderSet::simple(
                    rshader::shader_source!("shaders", "declarations.glsl", "grass.vert"),
                    rshader::shader_source!("shaders", "declarations.glsl", "pbr.glsl", "grass.frag"),
                )
                .unwrap(),
            }],
            vec![SingularLayerDesc {
                generate: ComputeShader::new(
                    rshader::shader_source!(
                        "shaders",
                        "declarations.glsl",
                        "hash.glsl",
                        "gen-grass-canopy.comp"
                    ),
                    "grass-canopy".to_string(),
                ),
                cache_size: 32,
                dependency_mask: LayerType::Normals.bit_mask(),
                level: VNode::LEVEL_CELL_1M,
                ty: SingularLayerType::GrassCanopy,
                texture_resolution: 516,
                texture_format: TextureFormat::RGBA8,
            }],
        );
        let quadtree =
            QuadTree::new(cache.tile_desc(LayerType::Displacements).texture_resolution - 1);

        let shader = rshader::ShaderSet::simple(
            rshader::shader_source!("shaders", "declarations.glsl", "terrain.vert"),
            rshader::shader_source!(
                "shaders",
                "declarations.glsl",
                "pbr.glsl",
                "terrain.frag"
            ),
        )
        .unwrap();

        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            size: std::mem::size_of::<GlobalUniformBlock>() as u64,
            usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::UNIFORM,
            label: Some("buffer.globals"),
            mapped_at_creation: false,
        });
        let node_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            size: (std::mem::size_of::<NodeState>() * 1024) as u64,
            usage: wgpu::BufferUsage::COPY_DST
                | wgpu::BufferUsage::UNIFORM
                | wgpu::BufferUsage::STORAGE,
            label: Some("buffer.nodes"),
            mapped_at_creation: false,
        });
        let index_buffer = quadtree.create_index_buffers(device);

        let noise = mapfile.read_texture(device, queue, "noise")?;
        let sky = mapfile.read_texture(device, queue, "sky")?;
        let transmittance = mapfile.read_texture(device, queue, "transmittance")?;
        let inscattering = mapfile.read_texture(device, queue, "inscattering")?;

        let bc4_staging = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d { width: 256, height: 256, depth_or_array_layers: 1 },
            format: wgpu::TextureFormat::Rg32Uint,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsage::COPY_SRC
                | wgpu::TextureUsage::COPY_DST
                | wgpu::TextureUsage::STORAGE
                | wgpu::TextureUsage::SAMPLED,
            label: Some("texture.staging.bc4"),
        });
        let bc5_staging = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d { width: 256, height: 256, depth_or_array_layers: 1 },
            format: wgpu::TextureFormat::Rgba32Uint,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsage::COPY_SRC
                | wgpu::TextureUsage::COPY_DST
                | wgpu::TextureUsage::STORAGE
                | wgpu::TextureUsage::SAMPLED,
            label: Some("texture.staging.bc5"),
        });

        let aerial_perspective = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d { width: 9, height: 9, depth_or_array_layers: 1024 },
            format: wgpu::TextureFormat::Rgba16Float,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsage::COPY_SRC
                | wgpu::TextureUsage::COPY_DST
                | wgpu::TextureUsage::STORAGE
                | wgpu::TextureUsage::SAMPLED,
            label: Some("texture.aerial_perspective"),
        });

        let gpu_state = GpuState {
            noise,
            sky,
            transmittance,
            inscattering,
            aerial_perspective,
            bc4_staging,
            bc5_staging,
            tile_cache: cache.make_gpu_tile_cache(device),
            mesh_cache: cache.make_gpu_mesh_cache(device),
            texture_cache: cache.make_gpu_texture_cache(device),
            globals,
            node_buffer,
        };

        let sky_shader = rshader::ShaderSet::simple(
            rshader::shader_source!("shaders", "declarations.glsl", "sky.vert"),
            rshader::shader_source!(
                "shaders",
                "declarations.glsl",
                "pbr.glsl",
                "sky.frag",
                "atmosphere.glsl"
            ),
        )
        .unwrap();

        let aerial_perspective = ComputeShader::new(
            rshader::shader_source!(
                "shaders",
                "declarations.glsl",
                "gen-aerial-perspective.comp",
                "atmosphere.glsl"
            ),
            "gen-aerial-perspective".to_string(),
        );

        Ok(Self {
            bindgroup_pipeline: None,
            shader,

            index_buffer,

            sky_shader,
            sky_bindgroup_pipeline: None,
            aerial_perspective,

            gpu_state,
            quadtree,
            mapfile,
            cache,
        })
    }

    fn loading_complete(&self) -> bool {
        VNode::roots().iter().copied().all(|root| {
            self.cache.tiles.contains(root, LayerType::Heightmaps)
                && self.cache.tiles.contains(root, LayerType::Albedo)
                && self.cache.tiles.contains(root, LayerType::Roughness)
        })
    }

    /// Returns whether initial map file streaming has completed for tiles in the vicinity of
    /// `camera`.
    ///
    /// Terra cannot render any terrain until all root tiles have been downloaded and streamed to
    /// the GPU. This function returns whether tohse tiles have been streamed, and also initiates
    /// streaming of more detailed tiles for the indicated tile position.
    pub fn poll_loading_status(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera: mint::Point3<f64>,
    ) -> bool {
        if !self.loading_complete() {
            self.cache.update(device, queue, &self.gpu_state, &self.mapfile, camera);
            self.loading_complete()
        } else {
            true
        }
    }

    /// Render the terrain.
    ///
    /// This function will block if the root tiles haven't been downloaded/loaded from disk. If
    /// you want to avoid this, call `poll_loading_status` first to see whether this function will
    /// block.
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_buffer: &wgpu::TextureView,
        depth_buffer: &wgpu::TextureView,
        _frame_size: (u32, u32),
        view_proj: mint::ColumnMatrix4<f32>,
        camera: mint::Point3<f64>,
    ) {
        if self.shader.refresh() {
            self.bindgroup_pipeline = None;
        }

        if self.bindgroup_pipeline.is_none() {
            let (bind_group, bind_group_layout) = self.gpu_state.bind_group_for_shader(
                device,
                &self.shader,
                hashmap![
                    "node".into() => (true, wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.gpu_state.node_buffer,
                        offset: 0,
                        size: Some((mem::size_of::<NodeState>() as u64).try_into().unwrap()),
                    }))
                ],
                HashMap::new(),
                "terrain",
            );
            let render_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                    label: Some("pipeline.terrain.layout"),
                });
            self.bindgroup_pipeline = Some((
                bind_group,
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    layout: Some(&render_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                            label: Some("shader.terrain.vertex"),
                            source: wgpu::ShaderSource::SpirV(self.shader.vertex().into()),
                            flags: wgpu::ShaderFlags::empty(),
                        }),
                        entry_point: "main",
                        buffers: &[],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                            label: Some("shader.terrain.fragment"),
                            source: wgpu::ShaderSource::SpirV(self.shader.fragment().into()),
                            flags: wgpu::ShaderFlags::empty(),
                        }),
                        entry_point: "main",
                        targets: &[wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Bgra8UnormSrgb,
                            blend: Some(wgpu::BlendState {
                                color: wgpu::BlendComponent::REPLACE,
                                alpha: wgpu::BlendComponent::REPLACE,
                            }),
                            write_mask: wgpu::ColorWrite::ALL,
                        }],
                    }),
                    primitive: wgpu::PrimitiveState {
                        cull_mode: Some(wgpu::Face::Front),
                        ..Default::default()
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::Greater,
                        bias: Default::default(),
                        stencil: Default::default(),
                    }),
                    multisample: Default::default(),
                    label: Some("pipeline.terrain"),
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
                HashMap::new(),
                HashMap::new(),
                "sky",
            );
            let render_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    bind_group_layouts: [&bind_group_layout][..].into(),
                    push_constant_ranges: &[],
                    label: Some("pipeline.sky.layout"),
                });
            self.sky_bindgroup_pipeline = Some((
                bind_group,
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    layout: Some(&render_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                            label: Some("shader.sky.vertex"),
                            source: wgpu::ShaderSource::SpirV(self.sky_shader.vertex().into()),
                            flags: wgpu::ShaderFlags::VALIDATION,
                        }),
                        entry_point: "main",
                        buffers: &[],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                            label: Some("shader.sky.fragment"),
                            source: wgpu::ShaderSource::SpirV(self.sky_shader.fragment().into()),
                            flags: wgpu::ShaderFlags::VALIDATION,
                        }),
                        entry_point: "main",
                        targets: &[wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Bgra8UnormSrgb,
                            blend: Some(wgpu::BlendState {
                                color: wgpu::BlendComponent::REPLACE,
                                alpha: wgpu::BlendComponent::REPLACE,
                            }),
                            write_mask: wgpu::ColorWrite::ALL,
                        }],
                    }),
                    primitive: Default::default(),
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_compare: wgpu::CompareFunction::GreaterEqual,
                        depth_write_enabled: false,
                        bias: Default::default(),
                        stencil: Default::default(),
                    }),
                    multisample: Default::default(),
                    label: Some("pipeline.sky"),
                }),
            ));
        }

        // Update the tile cache and then block until root tiles have been downloaded and streamed
        // to the GPU.
        self.cache.update(device, queue, &self.gpu_state, &self.mapfile, camera);
        while !self.poll_loading_status(device, queue, camera) {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        self.quadtree.update_visibility(&self.cache.tiles, camera);
        self.quadtree.prepare_vertex_buffer(
            queue,
            &mut self.gpu_state.node_buffer,
            &self.cache,
            camera,
        );

        queue.write_buffer(
            &self.gpu_state.globals,
            0,
            bytemuck::bytes_of(&GlobalUniformBlock {
                view_proj,
                view_proj_inverse: cgmath::Matrix4::from(view_proj).invert().unwrap().into(),
                camera: mint::Point3 { x: camera.x as f32, y: camera.y as f32, z: camera.z as f32 },
                sun_direction: mint::Vector3 { x: 0.4, y: 0.7, z: 0.2 },
                padding: [0.0; 2],
            }),
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("encoder.render"),
        });
        {
            self.aerial_perspective.refresh();
            self.aerial_perspective.run(
                device,
                &mut encoder,
                &self.gpu_state,
                (1, 1, self.quadtree.node_buffer_length() as u32),
                &0,
            );

            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: color_buffer,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }),
                        store: true,
                    },
                }],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_buffer,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
                label: Some("renderpass"),
            });
            rpass.set_pipeline(&self.bindgroup_pipeline.as_ref().unwrap().1);
            self.quadtree.render(
                &mut rpass,
                &self.index_buffer,
                &self.bindgroup_pipeline.as_ref().unwrap().0,
            );

            self.cache.render_meshes(device, &queue, &mut rpass, &self.gpu_state, camera);

            rpass.set_pipeline(&self.sky_bindgroup_pipeline.as_ref().unwrap().1);
            rpass.set_bind_group(0, &self.sky_bindgroup_pipeline.as_ref().unwrap().0, &[]);
            rpass.draw(0..3, 0..1);
        }

        queue.submit(Some(encoder.finish()));
    }

    pub fn get_height(&self, latitude: f64, longitude: f64) -> f32 {
        for level in (0..=VNode::LEVEL_CELL_1M).rev() {
            if let Some(height) = self.cache.tiles.get_height(latitude, longitude, level) {
                return height;
            }
        }
        0.0
    }
}
