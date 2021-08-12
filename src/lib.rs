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
mod types;
mod utils;

use crate::cache::{LayerType, MeshCacheDesc, MeshType};
use crate::generate::MapFileBuilder;
use crate::mapfile::MapFile;
use crate::terrain::quadtree::node::VNode;
use anyhow::Error;
use cache::UnifiedPriorityCache;
use cgmath::SquareMatrix;
use gpu_state::{GlobalUniformBlock, GpuState};
use std::array::IntoIter;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use terrain::quadtree::QuadTree;
use utils::math::InfiniteFrustum;
use wgpu::util::DeviceExt;

pub use crate::generate::BLUE_MARBLE_URLS;

pub struct Terrain {
    sky_shader: rshader::ShaderSet,
    sky_bindgroup_pipeline: Option<(wgpu::BindGroup, wgpu::RenderPipeline)>,
    // aerial_perspective: ComputeShader<u32>,
    gpu_state: GpuState,
    quadtree: QuadTree,
    mapfile: Arc<MapFile>,

    cache: UnifiedPriorityCache,
}
impl Terrain {
    pub async fn generate_and_new<P: AsRef<Path>, F: FnMut(&str, usize, usize) + Send>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dataset_directory: P,
        mut progress_callback: F,
    ) -> Result<Self, Error> {
        let mapfile = Arc::new(futures::executor::block_on(MapFileBuilder::new().build())?);

        let dataset_directory = dataset_directory.as_ref();
        generate::generate_heightmaps(
            &*mapfile,
            dataset_directory.join("ETOPO1_Ice_c_geotiff.zip"),
            dataset_directory.join("nasadem"),
            dataset_directory.join("nasadem_reprojected"),
            &mut progress_callback,
        )
        .await?;
        generate::generate_albedos(
            &*mapfile,
            dataset_directory.join("bluemarble"),
            &mut progress_callback,
        )
        .await?;
        generate::generate_roughness(&*mapfile, &mut progress_callback).await?;
        generate::generate_materials(
            &*mapfile,
            dataset_directory.join("free_pbr"),
            &mut progress_callback,
        )
        .await?;

        Self::new_impl(device, queue, mapfile)
    }

    /// Create a new Terrain object.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Self, Error> {
        let mapfile = Arc::new(futures::executor::block_on(MapFileBuilder::new().build())?);
        Self::new_impl(device, queue, mapfile)
    }

    fn new_impl(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mapfile: Arc<MapFile>,
    ) -> Result<Self, Error> {
        let cache = UnifiedPriorityCache::new(
            device,
            Arc::clone(&mapfile),
            vec![
                MeshCacheDesc {
                    ty: MeshType::Terrain,
                    max_bytes_per_node: 0,
                    entries_per_node: 4,
                    min_level: 0,
                    max_level: VNode::LEVEL_CELL_5MM,
                    index_format: wgpu::IndexFormat::Uint16,
                    index_buffer: QuadTree::create_index_buffer(device, 64),
                    render_overlapping_levels: false,
                    cull_mode: Some(wgpu::Face::Front),
                    render: rshader::ShaderSet::simple(
                        rshader::shader_source!("shaders", "terrain.vert", "declarations.glsl"),
                        rshader::shader_source!(
                            "shaders",
                            "terrain.frag",
                            "declarations.glsl",
                            "pbr.glsl"
                        ),
                    )
                    .unwrap(),
                },
                MeshCacheDesc {
                    ty: MeshType::Grass,
                    max_bytes_per_node: 128 * 128 * 64,
                    entries_per_node: 16,
                    min_level: VNode::LEVEL_SIDE_19M,
                    max_level: VNode::LEVEL_SIDE_5M,
                    cull_mode: None,
                    render_overlapping_levels: true,
                    index_format: wgpu::IndexFormat::Uint32,
                    index_buffer: {
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("buffer.index.grass"),
                            contents: bytemuck::cast_slice(
                                &*(0..32 * 32)
                                    .flat_map(|i| {
                                        IntoIter::new([
                                            0u32, 1, 2, 3, 2, 1, 2, 3, 4, 5, 4, 3, 4, 5, 6,
                                        ])
                                        .map(move |j| j + i * 7)
                                    })
                                    .collect::<Vec<u32>>(),
                            ),
                            usage: wgpu::BufferUsage::INDEX,
                        })
                    },
                    render: rshader::ShaderSet::simple(
                        rshader::shader_source!("shaders", "grass.vert", "declarations.glsl"),
                        rshader::shader_source!(
                            "shaders",
                            "grass.frag",
                            "declarations.glsl",
                            "pbr.glsl"
                        ),
                    )
                    .unwrap(),
                },
            ],
        );
        let gpu_state = GpuState::new(device, queue, &mapfile, &cache)?;
        let quadtree = QuadTree::new();

        let sky_shader = rshader::ShaderSet::simple(
            rshader::shader_source!("shaders", "sky.vert", "declarations.glsl"),
            rshader::shader_source!(
                "shaders",
                "sky.frag",
                "declarations.glsl",
                "pbr.glsl",
                "atmosphere.glsl"
            ),
        )
        .unwrap();
        // let aerial_perspective = ComputeShader::new(
        //     rshader::shader_source!(
        //         "shaders",
        //         "gen-aerial-perspective.comp",
        //         "declarations.glsl",
        //         "atmosphere.glsl"
        //     ),
        //     "gen-aerial-perspective".to_string(),
        // );

        Ok(Self {
            sky_shader,
            sky_bindgroup_pipeline: None,
            // aerial_perspective,
            gpu_state,
            quadtree,
            mapfile,
            cache,
        })
    }

    fn loading_complete(&self) -> bool {
        VNode::roots().iter().copied().all(|root| {
            self.cache.contains_all(
                root,
                LayerType::Heightmaps.bit_mask()
                    | LayerType::Albedo.bit_mask()
                    | LayerType::Roughness.bit_mask(),
            )
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
        self.quadtree.update_priorities(&self.cache, camera);
        if !self.loading_complete() {
            self.cache.update(
                device,
                queue,
                &self.gpu_state,
                &self.mapfile,
                &mut self.quadtree,
                camera,
            );
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

        let frustum = {
            let view_proj: cgmath::Matrix4<f64> =
                cgmath::Matrix4::<f32>::from(view_proj).cast().unwrap();
            let view_proj = view_proj
                * cgmath::Matrix4::from_translation(cgmath::Vector3::new(
                    -camera.x, -camera.y, -camera.z,
                ));
            InfiniteFrustum::from_matrix(view_proj)
        };
        let relative_frustum =
            InfiniteFrustum::from_matrix(cgmath::Matrix4::<f32>::from(view_proj).cast().unwrap());
        queue.write_buffer(
            &self.gpu_state.globals,
            0,
            bytemuck::bytes_of(&GlobalUniformBlock {
                view_proj,
                view_proj_inverse: cgmath::Matrix4::from(view_proj).invert().unwrap().into(),
                frustum_planes: [
                    relative_frustum.planes[0].cast().unwrap().into(),
                    relative_frustum.planes[1].cast().unwrap().into(),
                    relative_frustum.planes[2].cast().unwrap().into(),
                    relative_frustum.planes[3].cast().unwrap().into(),
                    relative_frustum.planes[4].cast().unwrap().into(),
                ],
                camera: [camera.x as f32, camera.y as f32, camera.z as f32, 0.0],
                sun_direction: [0.4, 0.7, 0.2, 0.0],
            }),
        );

        self.quadtree.update_priorities(&self.cache, camera);

        // Update the tile cache and then block until root tiles have been downloaded and streamed
        // to the GPU.
        self.cache.update(
            device,
            queue,
            &self.gpu_state,
            &self.mapfile,
            &mut self.quadtree,
            camera,
        );
        while !self.poll_loading_status(device, queue, camera) {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("encoder.render"),
        });

        {
            self.cache.cull_meshes(device, &mut encoder, &self.gpu_state);

            // self.aerial_perspective.refresh();
            // self.aerial_perspective.run(
            //     device,
            //     &mut encoder,
            //     &self.gpu_state,
            //     (1, 1, self.quadtree.node_buffer_length() as u32),
            //     &0,
            // );

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
            self.cache.render_meshes(device, &mut rpass, &self.gpu_state);

            rpass.set_pipeline(&self.sky_bindgroup_pipeline.as_ref().unwrap().1);
            rpass.set_bind_group(0, &self.sky_bindgroup_pipeline.as_ref().unwrap().0, &[]);
            rpass.draw(0..3, 0..1);
        }

        queue.submit(Some(encoder.finish()));
    }

    pub fn get_height(&self, latitude: f64, longitude: f64) -> f32 {
        for level in (0..=VNode::LEVEL_CELL_1M).rev() {
            if let Some(height) = self.cache.get_height(latitude, longitude, level) {
                return height;
            }
        }
        0.0
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn check_send() {
        struct Helper<T>(T);
        trait AssertImpl {
            fn assert() {}
        }
        impl<T: Send> AssertImpl for Helper<T> {}
        Helper::<super::Terrain>::assert();
    }
}
