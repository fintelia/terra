//! Terra is a large scale terrain generation and rendering library built on top of wgpu.
#![cfg_attr(test, feature(test))]

#[cfg(test)]
extern crate test;

#[macro_use]
extern crate lazy_static;

mod astro;
mod billboards;
mod cache;
mod compute_shader;
mod gpu_state;
mod mapfile;
mod speedtree_xml;
mod stream;

use crate::cache::MeshCacheDesc;
use crate::mapfile::MapFile;
use anyhow::Error;
use billboards::Models;
use cache::layer::{LayerType, MeshType};
use cache::TileCache;
use cgmath::{SquareMatrix, Vector3, Zero};
use compute_shader::ComputeShader;
use gpu_state::{GlobalUniformBlock, GpuState};
use std::collections::HashMap;
use std::sync::Arc;
use terra_types::{InfiniteFrustum, VNode};

pub const DEFAULT_TILE_SERVER_URL: &str = "https://terra2.fintelia.io/";

pub struct Terrain {
    sky_shader: rshader::ShaderSet,
    sky_bindgroup_pipeline: Option<(wgpu::BindGroup, wgpu::RenderPipeline)>,
    stars_shader: rshader::ShaderSet,
    stars_bindgroup_pipeline: Option<(wgpu::BindGroup, wgpu::RenderPipeline)>,
    gpu_state: GpuState,
    _mapfile: Arc<MapFile>,
    cache: TileCache,
    generate_skyview: ComputeShader<()>,
    view_proj: mint::ColumnMatrix4<f32>,
    shadow_view_proj: mint::ColumnMatrix4<f32>,
    camera: mint::Point3<f64>,
    sun_direction: Vector3<f32>,
    sidereal_time: f32,
    _models: Models,
}
impl Terrain {
    /// Create a new Terrain object.
    pub async fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        server: String,
    ) -> Result<Self, Error> {
        let mapfile = Arc::new(MapFile::new(server).await?);

        let mesh_layers = MeshType::iter()
            .map(|ty| match ty {
                MeshType::Terrain => MeshCacheDesc {
                    ty,
                    max_bytes_per_node: 0,
                    entries_per_node: 4,
                    min_level: 0,
                    max_level: VNode::LEVEL_CELL_5MM,
                    index_buffer: {
                        let mut data = Vec::new();
                        let resolution = 64;
                        let half_resolution = resolution / 2;
                        let width = resolution + 1;
                        for k in 0..2 {
                            for h in 0..2 {
                                for y in 0..half_resolution {
                                    for x in 0..half_resolution {
                                        for offset in [0, 1, width, 1, width + 1, width].iter() {
                                            data.push(
                                                offset
                                                    + ((h * half_resolution + x)
                                                        + (k * half_resolution + y) * width),
                                            );
                                        }
                                    }
                                }
                            }
                        }
                        data
                    },
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
                    render_shadow: None, /*Some(
                                             rshader::ShaderSet::simple(
                                                 rshader::shader_source!("shaders", "terrain.vert", "declarations.glsl"),
                                                 rshader::shader_source!("shaders", "shadowpass.frag"),
                                             )
                                             .unwrap(),
                                         )*/
                },
                MeshType::Grass => MeshCacheDesc {
                    ty,
                    max_bytes_per_node: 128 * 128 * 64,
                    entries_per_node: 16,
                    min_level: VNode::LEVEL_SIDE_19M,
                    max_level: VNode::LEVEL_SIDE_5M,
                    cull_mode: None,
                    render_overlapping_levels: true,
                    index_buffer: (0..32 * 32)
                        .flat_map(|i| {
                            IntoIterator::into_iter([
                                0u32, 1, 2, 3, 2, 1, 2, 3, 4, 5, 4, 3, 4, 5, 6,
                            ])
                            .map(move |j| j + i * 7)
                        })
                        .collect::<Vec<u32>>(),
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
                    render_shadow: None,
                },
                MeshType::TreeBillboards => MeshCacheDesc {
                    ty,
                    max_bytes_per_node: 128 * 128 * 64,
                    entries_per_node: 16,
                    min_level: VNode::LEVEL_SIDE_1KM,
                    max_level: VNode::LEVEL_SIDE_1KM,
                    cull_mode: None,
                    render_overlapping_levels: true,
                    index_buffer: (0..32 * 32)
                        .flat_map(|i| {
                            IntoIterator::into_iter([0u32, 1, 2, 3, 2, 1]).map(move |j| j + i * 4)
                        })
                        .collect::<Vec<u32>>(),
                    render: rshader::ShaderSet::simple(
                        rshader::shader_source!(
                            "shaders",
                            "tree-billboards.vert",
                            "declarations.glsl"
                        ),
                        rshader::shader_source!(
                            "shaders",
                            "tree-billboards.frag",
                            "declarations.glsl",
                            "pbr.glsl"
                        ),
                    )
                    .unwrap(),
                    render_shadow: None, /*Some(
                                             rshader::ShaderSet::simple(
                                                 rshader::shader_source!(
                                                     "shaders",
                                                     "tree-billboards.vert",
                                                     "declarations.glsl";
                                                     "SHADOWPASS" = "1"
                                                 ),
                                                 rshader::shader_source!(
                                                     "shaders",
                                                     "tree-billboards.frag",
                                                     "declarations.glsl",
                                                     "pbr.glsl";
                                                     "SHADOWPASS" = "1"
                                                 ),
                                             )
                                             .unwrap(),
                                         )*/
                },
            })
            .collect();

        let models = Models::new(&mapfile).await?;
        let cache = TileCache::new(device, Arc::clone(&mapfile), mesh_layers);
        let gpu_state = GpuState::new(device, queue, &mapfile, &cache, &models).await?;

        models.render_billboards(device, queue, &gpu_state);

        let sky_shader = rshader::ShaderSet::simple(
            rshader::shader_source!("shaders", "sky.vert", "declarations.glsl"),
            rshader::shader_source!(
                "shaders",
                "sky.frag",
                "declarations.glsl",
                "pbr.glsl",
                "atmosphere.glsl",
                "hash.glsl"
            ),
        )
        .unwrap();

        let stars_shader = rshader::ShaderSet::simple(
            rshader::shader_source!("shaders", "stars.vert", "declarations.glsl"),
            rshader::shader_source!(
                "shaders",
                "stars.frag",
                "declarations.glsl",
                "pbr.glsl",
                "atmosphere.glsl"
            ),
        )
        .unwrap();

        let generate_skyview = ComputeShader::new(
            rshader::shader_source!(
                "shaders",
                "gen-skyview.comp",
                "declarations.glsl",
                "atmosphere.glsl"
            ),
            "gen-skyview".to_string(),
        );

        Ok(Self {
            sky_shader,
            sky_bindgroup_pipeline: None,
            stars_shader,
            stars_bindgroup_pipeline: None,
            gpu_state,
            _mapfile: mapfile,
            cache,
            generate_skyview,
            view_proj: cgmath::Matrix4::zero().into(),
            shadow_view_proj: cgmath::Matrix4::zero().into(),
            camera: mint::Point3::from_slice(&[0.0, 0.0, 0.0]),
            sun_direction: cgmath::Vector3::new(0.4, 0.7, 0.2),
            sidereal_time: 0.0,
            _models: models,
        })
    }

    /// Continuously polls until file streaming has completed for tiles in the vicinity of
    /// `camera`.
    ///
    /// Terra cannot render any terrain until all root tiles have been downloaded and streamed to
    /// the GPU. This function returns whether those tiles have been streamed, and also initiates
    /// streaming of more detailed tiles for the indicated camera position.
    pub fn poll_loading_status<F: FnMut(f32)>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera: mint::Point3<f64>,
        progress_callback: F,
    ) {
        self.cache.wait_for_uploads(device, queue, &self.gpu_state, camera, progress_callback)
    }

    /// Update the terrain.
    ///
    /// This function will block if the root tiles haven't been downloaded/loaded from disk. If
    /// you want to avoid this, call `poll_loading_status` first to see whether this function will
    /// block.
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view_proj: mint::ColumnMatrix4<f32>,
        camera: mint::Point3<f64>,
        julian_day: f64,
    ) {
        self.view_proj = view_proj;
        let shadow_view = cgmath::Matrix4::look_to_rh(
            cgmath::Point3::new(0., 0., 0.),
            cgmath::Vector3::new(-0.4, -0.7, -0.2),
            cgmath::Vector3::unit_z(),
        );
        let shadow_proj = cgmath::Matrix4::new(
            1.0 / 8192.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0 / 8192.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.0 / 102400.0,
            0.0,
            0.0,
            0.0,
            0.5,
            1.0,
        );
        self.shadow_view_proj = (shadow_proj * shadow_view).into();
        self.camera = camera;

        if self._models.refresh() {
            self._models.render_billboards(device, queue, &self.gpu_state);
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
                        module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                            label: Some("shader.sky.vertex"),
                            source: self.sky_shader.vertex(),
                        }),
                        entry_point: "main",
                        buffers: &[],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                            label: Some("shader.sky.fragment"),
                            source: self.sky_shader.fragment(),
                        }),
                        entry_point: "main",
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Bgra8UnormSrgb,
                            blend: Some(wgpu::BlendState {
                                color: wgpu::BlendComponent::REPLACE,
                                alpha: wgpu::BlendComponent::REPLACE,
                            }),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
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
                    multiview: None,
                    label: Some("pipeline.sky"),
                }),
            ));
        }

        if self.stars_shader.refresh() {
            self.stars_bindgroup_pipeline = None;
        }
        if self.stars_bindgroup_pipeline.is_none() {
            let (bind_group, bind_group_layout) = self.gpu_state.bind_group_for_shader(
                device,
                &self.stars_shader,
                HashMap::new(),
                HashMap::new(),
                "stars",
            );
            let render_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    bind_group_layouts: [&bind_group_layout][..].into(),
                    push_constant_ranges: &[],
                    label: Some("pipeline.stars.layout"),
                });
            self.stars_bindgroup_pipeline = Some((
                bind_group,
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    layout: Some(&render_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                            label: Some("shader.stars.vertex"),
                            source: self.stars_shader.vertex(),
                        }),
                        entry_point: "main",
                        buffers: &[],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                            label: Some("shader.stars.fragment"),
                            source: self.stars_shader.fragment(),
                        }),
                        entry_point: "main",
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Bgra8UnormSrgb,
                            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
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
                    multiview: None,
                    label: Some("pipeline.stars"),
                }),
            ));
        }

        self.cache.update(device, queue, &self.gpu_state, camera);

        // Block until root tiles have been downloaded and streamed to the GPU.
        while !VNode::roots().iter().copied().all(|root| {
            self.cache.contains_layers(
                root,
                LayerType::BaseHeightmaps.bit_mask() | LayerType::BaseAlbedo.bit_mask(),
            )
        }) {
            std::thread::sleep(std::time::Duration::from_millis(10));
            self.cache.update(device, queue, &self.gpu_state, camera);
        }

        self.generate_skyview.refresh(device, &self.gpu_state);
        self.cache.update_meshes(device, &self.gpu_state);

        let sidereal_time = astro::mn_sidr(julian_day);
        self.sun_direction = {
            let n = julian_day - 2451545.0;
            let l: f64 = (280.460 + 0.9856474 * n).to_radians();
            let g: f64 = (357.528 + 0.9856003 * n).to_radians();
            let oblq_eclip = (23.439 - 0.0000004 * n).to_radians();
            let lambda = l + 1.915 * f64::sin(g) + 0.02 * f64::sin(2.0 * g);
            let declination = astro::dec_frm_ecl(lambda, 0.0, oblq_eclip);
            let ascension = astro::asc_frm_ecl(lambda, 0.0, oblq_eclip);
            cgmath::Vector3::new(
                f64::cos(declination) * f64::cos(ascension - sidereal_time),
                f64::cos(declination) * f64::sin(ascension - sidereal_time),
                f64::sin(declination),
            )
            .cast()
            .unwrap()
        };
        self.sidereal_time = sidereal_time as f32;
    }

    pub fn render_shadows(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let relative_frustum = InfiniteFrustum::from_matrix(
            cgmath::Matrix4::<f32>::from(self.shadow_view_proj).cast().unwrap(),
        );
        queue.write_buffer(
            &self.gpu_state.globals,
            0,
            bytemuck::bytes_of(&GlobalUniformBlock {
                view_proj: self.shadow_view_proj,
                view_proj_inverse: cgmath::Matrix4::from(self.shadow_view_proj)
                    .invert()
                    .unwrap()
                    .into(),
                frustum_planes: [
                    relative_frustum.planes[0].cast().unwrap().into(),
                    relative_frustum.planes[1].cast().unwrap().into(),
                    relative_frustum.planes[2].cast().unwrap().into(),
                    relative_frustum.planes[3].cast().unwrap().into(),
                    relative_frustum.planes[4].cast().unwrap().into(),
                ],
                shadow_view_proj: self.shadow_view_proj,
                camera: [self.camera.x as f32, self.camera.y as f32, self.camera.z as f32],
                screen_width: 2048.0,
                sun_direction: self.sun_direction.into(),
                screen_height: 2048.0,
                sidereal_time: self.sidereal_time,
                exposure: 1.0,
                _padding: [0.0; 2],
            }),
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("encoder.render"),
        });

        {
            self.cache.cull_meshes(device, &mut encoder, &self.gpu_state);

            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.gpu_state.shadowmap.1,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
                label: Some("shadowpass"),
            });
            self.cache.render_mesh_shadows(device, &mut rpass, &self.gpu_state);
        }

        queue.submit(Some(encoder.finish()));
    }

    /// Render the terrain.
    ///
    /// Terrain::update must be called first.
    pub fn render(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_buffer: &wgpu::TextureView,
        depth_buffer: &wgpu::TextureView,
        frame_size: (u32, u32),
        render_view_proj: mint::ColumnMatrix4<f32>,
    ) {
        let relative_frustum = InfiniteFrustum::from_matrix(
            cgmath::Matrix4::<f32>::from(self.view_proj).cast().unwrap(),
        );
        queue.write_buffer(
            &self.gpu_state.globals,
            0,
            bytemuck::bytes_of(&GlobalUniformBlock {
                view_proj: render_view_proj,
                view_proj_inverse: cgmath::Matrix4::from(render_view_proj).invert().unwrap().into(),
                shadow_view_proj: self.shadow_view_proj,
                frustum_planes: [
                    relative_frustum.planes[0].cast().unwrap().into(),
                    relative_frustum.planes[1].cast().unwrap().into(),
                    relative_frustum.planes[2].cast().unwrap().into(),
                    relative_frustum.planes[3].cast().unwrap().into(),
                    relative_frustum.planes[4].cast().unwrap().into(),
                ],
                camera: [self.camera.x as f32, self.camera.y as f32, self.camera.z as f32],
                screen_width: frame_size.0 as f32,
                sun_direction: self.sun_direction.into(),
                screen_height: frame_size.1 as f32,
                sidereal_time: self.sidereal_time,
                exposure: 1.0 / (f32::powf(2.0, 17.0) * 1.2),
                _padding: [0.0; 2],
            }),
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("encoder.render"),
        });

        {
            self.cache.run_dynamic_generators(queue, &mut encoder, &self.gpu_state);
            self.cache.cull_meshes(device, &mut encoder, &self.gpu_state);

            self.generate_skyview.run(device, &mut encoder, &self.gpu_state, (16, 16, 1), &());

            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_buffer,
                    resolve_target: None,
                    ops: wgpu::Operations::default(),
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_buffer,
                    depth_ops: Some(wgpu::Operations::default()),
                    stencil_ops: None,
                }),
                label: Some("renderpass"),
            });
            self.cache.render_meshes(device, &mut rpass, &self.gpu_state);

            rpass.set_pipeline(&self.sky_bindgroup_pipeline.as_ref().unwrap().1);
            rpass.set_bind_group(0, &self.sky_bindgroup_pipeline.as_ref().unwrap().0, &[]);
            rpass.draw(0..3, 0..1);

            rpass.set_pipeline(&self.stars_bindgroup_pipeline.as_ref().unwrap().1);
            rpass.set_bind_group(0, &self.stars_bindgroup_pipeline.as_ref().unwrap().0, &[]);
            rpass.draw(0..9096 * 6, 0..1);
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
