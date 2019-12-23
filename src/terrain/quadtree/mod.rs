use astro::{coords, sun};
use byteorder::{LittleEndian, ReadBytesExt};
use cgmath::*;
use collision::{Frustum, Relation};
use failure::Error;
use memmap::Mmap;
use vec_map::VecMap;

use std::collections::VecDeque;
use std::sync::Arc;

use crate::coordinates::CoordinateSystem;
use crate::terrain::tile_cache::{LayerType, Priority, TileCache, TileHeader};

pub(crate) mod id;
pub(crate) mod node;
pub(crate) mod render;

pub(crate) use crate::terrain::quadtree::id::*;
pub(crate) use crate::terrain::quadtree::node::*;
pub(crate) use crate::terrain::quadtree::render::*;

/// The central object in terra. It holds all relevant state and provides functions to update and
/// render the terrain.
pub struct QuadTree {
    /// List of nodes in the `QuadTree`. The root is always at index 0.
    nodes: Vec<Node>,
    // ocean: Ocean<R>,

    // #[allow(unused)]
    // atmosphere: Atmosphere<R>,
    /// List of nodes that will be rendered.
    visible_nodes: Vec<NodeId>,
    partially_visible_nodes: Vec<(NodeId, u8)>,

    /// Cache holding nearby tiles for each layer.
    tile_cache_layers: VecMap<TileCache>,

    // index_buffer: Escape<Buffer<B>>,
    // index_buffer_partial: Escape<Buffer<B>>,

    // factory: F,
    // pso: gfx::PipelineState<R, pipe::Meta>,
    // pipeline_data: pipe::Data<R>,
    // sky_pso: gfx::PipelineState<R, sky_pipe::Meta>,
    // sky_pipeline_data: sky_pipe::Data<R>,
    // planet_mesh_pso: gfx::PipelineState<R, planet_mesh_pipe::Meta>,
    // planet_mesh_pipeline_data: planet_mesh_pipe::Data<R>,
    // instanced_mesh_pso: gfx::PipelineState<R, instanced_mesh_pipe::Meta>,
    // instanced_mesh_pipeline_data: instanced_mesh_pipe::Data<R>,
    // shaders_watcher: rshader::ShaderDirectoryWatcher,
    // shader: rshader::Shader<R>,
    // sky_shader: rshader::Shader<R>,
    // planet_mesh_shader: rshader::Shader<R>,
    // instanced_mesh_shader: rshader::Shader<R>,
    // num_planet_mesh_vertices: usize,
    node_states: Vec<NodeState>,
    // _materials: MaterialSet<R>,
    system: CoordinateSystem,
}

impl std::fmt::Debug for QuadTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QuadTree")
    }
}

#[allow(unused)]
impl QuadTree {
    pub(crate) fn new(
        header: TileHeader,
        data_file: Mmap,
        // materials: MaterialSet<R>,
        // sky: Skybox<R>,
        // factory: &mut Factory<B>,
        // encoder: &mut gfx::Encoder<R, C>,
        // color_buffer: &gfx::handle::RenderTargetView<R, gfx::format::Rgba16F>,
        // depth_buffer: &gfx::handle::DepthStencilView<R, gfx::format::Depth32F>,
        // mut context: AssetLoadContext,
    ) -> Result<Self, Error> {
        eprintln!("nodes.len() = {}", header.nodes.len());
        eprintln!("layers.len() = {}", header.layers.len());
        eprintln!(
            "heights = {} ({:?})",
            header.layers[0].tile_locations.len(),
            header.layers[0].payload_type
        );
        eprintln!(
            "colors = {} ({:?})",
            header.layers[1].tile_locations.len(),
            header.layers[1].payload_type
        );
        eprintln!(
            "normals = {} ({:?})",
            header.layers[2].tile_locations.len(),
            header.layers[2].payload_type
        );
        eprintln!(
            "splats = {} ({:?})",
            header.layers[3].tile_locations.len(),
            header.layers[3].payload_type
        );

        eprintln!();

        let world_size = 4194304.0;
        for spacing in -2..=1 {
            for radius in 2..10 {
                let max_level = (22i32 - (129u32 - 1).trailing_zeros() as i32 - spacing) as u8;
                let nodes = Node::make_nodes(world_size, radius as f32 * 1000.0, max_level);
                let hn = nodes.len();
                let dn = nodes.iter().filter(|n| n.level <= max_level - 2).count();
                let hmemory = (hn * (129 * 129 * 4)) as f32 / 2.0f32.powi(20);
                let memory = (hn * (129 * 129 * 4) + dn * (512 * 512 * 2)) as f32 / 2.0f32.powi(20);
                eprintln!(
                    "spacing={}m, radius={}km, hn={}, dn={} hmemory={}MiB, memory={}MiB",
                    2.0f32.powi(spacing),
                    radius,
                    hn,
                    dn,
                    hmemory,
                    memory
                );
            }
            eprintln!();
        }

        // let mut shaders_watcher = rshader::ShaderDirectoryWatcher::new(
        //     env::var("TERRA_SHADER_DIRECTORY").unwrap_or("src/shaders/glsl".to_string()),
        // )?;

        // let shader = rshader::Shader::simple(
        //     &mut factory,
        //     &mut shaders_watcher,
        //     shader_source!(
        //         "../../shaders/glsl",
        //         "version",
        //         "atmosphere",
        //         "terrain.glslv"
        //     ),
        //     shader_source!(
        //         "../../shaders/glsl",
        //         "version",
        //         "atmosphere",
        //         "hash",
        //         "terrain.glslf"
        //     ),
        // )?;

        // let sky_shader = rshader::Shader::simple(
        //     &mut factory,
        //     &mut shaders_watcher,
        //     shader_source!("../../shaders/glsl", "version", "sky.glslv"),
        //     shader_source!(
        //         "../../shaders/glsl",
        //         "version",
        //         "atmosphere",
        //         "hash",
        //         "sky.glslf"
        //     ),
        // )?;

        // let planet_mesh_shader = rshader::Shader::simple(
        //     &mut factory,
        //     &mut shaders_watcher,
        //     shader_source!("../../shaders/glsl", "version", "planet_mesh.glslv"),
        //     shader_source!(
        //         "../../shaders/glsl",
        //         "version",
        //         "atmosphere",
        //         "hash",
        //         "planet_mesh.glslf"
        //     ),
        // )?;

        // let instanced_mesh_shader = rshader::Shader::simple(
        //     &mut factory,
        //     &mut shaders_watcher,
        //     shader_source!("../../shaders/glsl", "version", "mesh.glslv"),
        //     shader_source!(
        //         "../../shaders/glsl",
        //         "version",
        //         "atmosphere",
        //         "hash",
        //         "mesh.glslf"
        //     ),
        // )?;

        let mut data_view = Arc::new(data_file);
        let mut tile_cache_layers = VecMap::new();
        for layer in header.layers.iter().cloned() {
            tile_cache_layers.insert(
                layer.layer_type as usize,
                TileCache::new(layer, data_view.clone()),
            );
        }

        // let noise_data = &data_view[header.noise.offset..][..header.noise.bytes];
        // let (noise_texture, noise_texture_view) = factory.create_texture_immutable_u8::<(
        //     gfx_core::format::R8_G8_B8_A8,
        //     gfx_core::format::Unorm,
        // )>(
        //     gfx::texture::Kind::D2(
        //         header.noise.resolution as u16,
        //         header.noise.resolution as u16,
        //         gfx::texture::AaMode::Single,
        //     ),
        //     gfx::texture::Mipmap::Allocated,
        //     &[gfx::memory::cast_slice(noise_data)],
        // )?;
        // encoder.generate_mipmap(&noise_texture_view);

        // let planet_mesh_data = &data_view[header.planet_mesh.offset..][..header.planet_mesh.bytes];
        // let planet_mesh_vertices = gfx::memory::cast_slice(planet_mesh_data);

        // let pm_texture_start = header.planet_mesh_texture.offset;
        // let pm_texture_end = pm_texture_start + header.planet_mesh_texture.bytes;
        // let pm_texture_data = &data_view[pm_texture_start..pm_texture_end];
        // let (planet_mesh_texture, planet_mesh_texture_view) = factory
        //     .create_texture_immutable_u8::<(gfx_core::format::R8_G8_B8_A8, gfx_core::format::Srgb)>(
        //         gfx::texture::Kind::D2(
        //             header.planet_mesh_texture.resolution as u16,
        //             header.planet_mesh_texture.resolution as u16,
        //             gfx::texture::AaMode::Single,
        //         ),
        //         gfx::texture::Mipmap::Allocated,
        //         &[gfx::memory::cast_slice(pm_texture_data)],
        //     )?;
        // encoder.generate_mipmap(&planet_mesh_texture_view);

        // let (
        //     foliage_mesh_offset,
        //     foliage_mesh_bytes,
        //     foliage_texture_offset,
        //     foliage_texture_bytes,
        //     foliage_texture_resolution,
        // ) = if let PayloadType::InstancedMesh {
        //     mesh: MeshDescriptor { offset, bytes, .. },
        //     texture:
        //         TextureDescriptor {
        //             offset: toffset,
        //             bytes: tbytes,
        //             format,
        //             resolution,
        //         },
        //     ..
        // } = header.layers[LayerType::Foliage as usize].payload_type
        // {
        //     assert_eq!(format, TextureFormat::SRGBA);
        //     (offset, bytes, toffset, tbytes, resolution)
        // } else {
        //     unreachable!()
        // };

        // let instanced_mesh_vertices =
        //     gfx::memory::cast_slice(&data_view[foliage_mesh_offset..][..foliage_mesh_bytes]);

        // let instanced_mesh_texture_data =
        //     gfx::memory::cast_slice(&data_view[foliage_texture_offset..][..foliage_texture_bytes]);
        // let (instanced_mesh_texture, instanced_mesh_texture_view) = factory
        //     .create_texture_immutable_u8::<(gfx_core::format::R8_G8_B8_A8, gfx_core::format::Srgb)>(
        //         gfx::texture::Kind::D2(
        //             foliage_texture_resolution as u16,
        //             foliage_texture_resolution as u16,
        //             gfx::texture::AaMode::Single,
        //         ),
        //         gfx::texture::Mipmap::Allocated,
        //         &[instanced_mesh_texture_data],
        //     )?;
        // encoder.generate_mipmap(&instanced_mesh_texture_view);

        // let heights_texture_view = tile_cache_layers[LayerType::Heights.index()]
        //     .get_texture_view_f32()
        //     .unwrap()
        //     .clone();
        // let colors_texture_view = tile_cache_layers[LayerType::Colors.index()]
        //     .get_texture_view_srgba()
        //     .unwrap()
        //     .clone();
        // let normals_texture_view = tile_cache_layers[LayerType::Normals.index()]
        //     .get_texture_view_rgba8()
        //     .unwrap()
        //     .clone();
        // let splats_texture_view = tile_cache_layers[LayerType::Splats.index()]
        //     .get_texture_view_r8()
        //     .unwrap()
        //     .clone();

        // let ocean = Ocean::new(&mut factory);
        // let atmosphere = Atmosphere::new(&mut factory, &mut context)?;

        // let sampler = factory.create_sampler(gfx::texture::SamplerInfo::new(
        //     gfx::texture::FilterMethod::Trilinear,
        //     gfx::texture::WrapMode::Clamp,
        // ));

        // let sampler_wrap = factory.create_sampler(gfx::texture::SamplerInfo::new(
        //     gfx::texture::FilterMethod::Trilinear,
        //     gfx::texture::WrapMode::Tile,
        // ));

        // Extra scope to work around lack of non-lexical lifetimes.

        // let transmittance = (
        //     atmosphere.transmittance.texture_view.clone(),
        //     sampler.clone(),
        // );
        // let inscattering = (
        //     atmosphere.inscattering.texture_view.clone(),
        //     sampler.clone(),
        // );

        let ww: Matrix4<f32> = header.system.world_to_warped_matrix().cast().unwrap();
        let world_to_warped = [
            [ww.x.x, ww.x.y, ww.x.z, ww.x.w],
            [ww.y.x, ww.y.y, ww.y.z, ww.y.w],
            [ww.z.x, ww.z.y, ww.z.z, ww.z.w],
            [ww.w.x, ww.w.y, ww.w.z, ww.w.w],
        ];

        Ok(Self {
            visible_nodes: Vec::new(),
            partially_visible_nodes: Vec::new(),
            // index_buffer,
            // index_buffer_partial,
            // pso: Self::make_pso(&mut factory, shader.as_shader_set())?,
            // pipeline_data: pipe::Data {
            //     instances: factory.create_constant_buffer::<NodeState>(header.nodes.len() * 3),
            //     model_view_projection: [[0.0; 4]; 4],
            //     camera_position: [0.0, 0.0, 0.0],
            //     sun_direction: [0.0, 0.70710678118, 0.70710678118],
            //     resolution: 0,
            //     world_to_warped: world_to_warped.clone(),
            //     heights: (heights_texture_view, sampler.clone()),
            //     colors: (colors_texture_view, sampler.clone()),
            //     normals: (normals_texture_view, sampler.clone()),
            //     splats: (splats_texture_view, sampler.clone()),
            //     materials: (materials.texture_view.clone(), sampler_wrap.clone()),
            //     sky: (sky.texture_view.clone(), sampler.clone()),
            //     ocean_surface: (ocean.texture_view.clone(), sampler_wrap.clone()),
            //     noise: (noise_texture_view, sampler_wrap),
            //     noise_wavelength: header.noise.wavelength,
            //     planet_radius: 6371000.0,
            //     atmosphere_radius: 6471000.0,
            //     transmittance: transmittance.clone(),
            //     inscattering: inscattering.clone(),
            //     color_buffer: color_buffer.clone(),
            //     depth_buffer: depth_buffer.clone(),
            // },
            // sky_pso: Self::make_sky_pso(&mut factory, sky_shader.as_shader_set())?,
            // sky_pipeline_data: sky_pipe::Data {
            //     ray_bottom_left: [0.0, 0.0, 0.0],
            //     ray_bottom_right: [0.0, 0.0, 0.0],
            //     ray_top_left: [0.0, 0.0, 0.0],
            //     ray_top_right: [0.0, 0.0, 0.0],
            //     camera_position: [0.0, 0.0, 0.0],
            //     sun_direction: [0.0, 0.70710678118, 0.70710678118],
            //     world_to_warped: world_to_warped.clone(),
            //     sky: (sky.texture_view.clone(), sampler.clone()),
            //     planet_radius: 6371000.0,
            //     atmosphere_radius: 6471000.0,
            //     transmittance: transmittance.clone(),
            //     inscattering: inscattering.clone(),
            //     color_buffer: color_buffer.clone(),
            //     depth_buffer: depth_buffer.clone(),
            // },
            // planet_mesh_pso: Self::make_planet_mesh_pso(
            //     &mut factory,
            //     planet_mesh_shader.as_shader_set(),
            // )?,
            // planet_mesh_pipeline_data: planet_mesh_pipe::Data {
            //     vertices: factory.create_vertex_buffer(planet_mesh_vertices),
            //     model_view_projection: [[0.0; 4]; 4],
            //     camera_position: [0.0, 0.0, 0.0],
            //     sun_direction: [0.0, 0.70710678118, 0.70710678118],
            //     planet_radius: 6371000.0,
            //     atmosphere_radius: 6471000.0,
            //     world_to_warped: world_to_warped.clone(),
            //     transmittance: transmittance.clone(),
            //     inscattering: inscattering.clone(),
            //     color: (planet_mesh_texture_view, sampler.clone()),
            //     color_buffer: color_buffer.clone(),
            //     depth_buffer: depth_buffer.clone(),
            // },
            // instanced_mesh_pso: Self::make_instanced_mesh_pso(
            //     &mut factory,
            //     instanced_mesh_shader.as_shader_set(),
            // )?,
            // instanced_mesh_pipeline_data: instanced_mesh_pipe::Data {
            //     vertices: factory.create_vertex_buffer(instanced_mesh_vertices),
            //     instances: tile_cache_layers[LayerType::Foliage.index()]
            //         .get_buffer()
            //         .unwrap()
            //         .clone(),
            //     model_view_projection: [[0.0; 4]; 4],
            //     camera_position: [0.0, 0.0, 0.0],
            //     sun_direction: [0.0, 0.70710678118, 0.70710678118],
            //     planet_radius: 6371000.0,
            //     atmosphere_radius: 6471000.0,
            //     world_to_warped: world_to_warped.clone(),
            //     albedo: (instanced_mesh_texture_view, sampler.clone()),
            //     transmittance: transmittance,
            //     inscattering: inscattering,
            //     color_buffer: color_buffer.clone(),
            //     depth_buffer: depth_buffer.clone(),
            // },
            // ocean,
            // atmosphere,
            // factory,
            // shaders_watcher,
            // shader,
            // sky_shader,
            // planet_mesh_shader,
            // instanced_mesh_shader,
            // num_planet_mesh_vertices: header.planet_mesh.num_vertices,
            nodes: header.nodes,
            node_states: Vec::new(),
            tile_cache_layers,
            // _materials: materials,
            system: header.system,
        })
    }

    // pub(crate) fn create_index_buffers<B: Backend>(
    //     &self,
    //     factory: &mut Factory<B>,
    //     queue: QueueId,
    // ) -> Result<(Escape<Buffer<B>>, Escape<Buffer<B>>), Error> {
    //     let make_index_buffer = |resolution: u16| -> Result<Escape<Buffer<B>>, Error> {
    //         let width = resolution + 1;
    //         let mut indices: Vec<u16> = Vec::new();
    //         for y in 0..resolution {
    //             for x in 0..resolution {
    //                 for offset in [0, 1, width, 1, width + 1, width].into_iter() {
    //                     indices.push(offset + (x + y * width));
    //                 }
    //             }
    //         }

    //         let index_buffer = factory.create_buffer(
    //             BufferInfo {
    //                 size: (2 * indices.len()) as u64,
    //                 usage: Usage::INDEX | Usage::TRANSFER_DST,
    //             },
    //             memory::Data,
    //         )?;

    //         unsafe {
    //             factory.upload_buffer(
    //                 &index_buffer,
    //                 0,
    //                 &indices,
    //                 None,
    //                 BufferState::new(queue).with_access(Access::INDEX_BUFFER_READ),
    //             );
    //         }

    //         Ok(index_buffer)
    //     };
    //     let resolution =
    //         (self.tile_cache_layers[LayerType::Heights.index()].resolution() - 1) as u16;

    //     Ok((
    //         make_index_buffer(resolution)?,
    //         make_index_buffer(resolution / 2)?,
    //     ))
    // }

    fn update_priorities(&mut self, camera: Point3<f32>) {
        for node in self.nodes.iter_mut() {
            node.priority = Priority::from_f32(
                (node.min_distance * node.min_distance)
                    / node.bounds.square_distance(camera).max(0.001),
            );
        }
        for (_, ref mut cache_layer) in self.tile_cache_layers.iter_mut() {
            cache_layer.update_priorities(&mut self.nodes);
        }
    }

    // fn update_cache<C: gfx_core::command::Buffer<R>>(&mut self, encoder: &mut gfx::Encoder<R, C>) {
    //     self.breadth_first(|qt, id| {
    //         if qt.nodes[id].priority < Priority::cutoff() {
    //             return false;
    //         }

    //         for layer in 0..NUM_LAYERS {
    //             if qt.nodes[id].tile_indices[layer].is_some()
    //                 && !qt.tile_cache_layers[layer].contains(id)
    //             {
    //                 qt.tile_cache_layers[layer].add_missing((qt.nodes[id].priority, id));
    //             }
    //         }
    //         true
    //     });
    //     for (_, ref mut cache_layer) in self.tile_cache_layers.iter_mut() {
    //         cache_layer.load_missing(&mut self.nodes, encoder);
    //     }
    // }

    fn update_visibility(&mut self, cull_frustum: Option<Frustum<f32>>) {
        self.visible_nodes.clear();
        self.partially_visible_nodes.clear();
        for node in self.nodes.iter_mut() {
            node.visible = false;
        }
        // Any node with all needed layers in cache is visible...
        self.breadth_first(|qt, id| {
            qt.nodes[id].visible =
                id == NodeId::root() || qt.nodes[id].priority >= Priority::cutoff();
            qt.nodes[id].visible
        });
        // ...Except if all its children are visible instead.
        self.breadth_first(|qt, id| {
            if qt.nodes[id].visible {
                let mut mask = 0;
                let mut has_visible_children = false;
                for (i, c) in qt.nodes[id].children.iter().enumerate() {
                    if c.is_none() || !qt.nodes[c.unwrap()].visible {
                        mask = mask | (1 << i);
                    }

                    if c.is_some() && qt.nodes[c.unwrap()].visible {
                        has_visible_children = true;
                    }
                }

                if let Some(frustum) = cull_frustum {
                    // TODO: Also try to cull parts of a node, if contains() returns Relation::Cross.
                    if frustum.contains(&qt.nodes[id].bounds.as_aabb3()) == Relation::Out {
                        mask = 0;
                    }
                }

                if mask == 0 {
                    qt.nodes[id].visible = false;
                } else if has_visible_children {
                    qt.partially_visible_nodes.push((id, mask));
                } else {
                    qt.visible_nodes.push(id);
                }

                true
            } else {
                false
            }
        });
    }

    pub fn update(
        &mut self,
        // mvp_mat: Matrix4<f32>,
        camera: Point3<f32>,
        cull_frustum: Option<Frustum<f32>>,
        // encoder: &mut gfx::Encoder<R, C>,
        // dt: f32,
    ) {
        let sun_direction = {
            let (ecl, distance_au) = sun::geocent_ecl_pos(180.0);
            let distance = distance_au * 149597870700.0;

            let e = 0.40905;
            let declination = coords::dec_frm_ecl(ecl.long, ecl.lat, e);
            let right_ascension = coords::asc_frm_ecl(ecl.long, ecl.lat, e);

            let eq_rect = Vector3::new(
                distance * declination.cos() * right_ascension.cos(),
                distance * declination.cos() * right_ascension.sin(),
                distance * declination.sin(),
            );

            // TODO: Is this conversion from equatorial coordinates to ECEF actually valid?
            let ecef = Vector3::new(eq_rect.x, -eq_rect.y, eq_rect.z);

            let world = self.system.ecef_to_world(ecef);
            let direction = world.normalize();
            [direction.x as f32, direction.y as f32, direction.z as f32]
        };

        // Convert the MVP matrix to "vecmath encoding".
        // let to_array = |v: Vector4<f32>| [v.x, v.y, v.z, v.w];
        // let mvp_mat = [
        //     to_array(mvp_mat.x),
        //     to_array(mvp_mat.y),
        //     to_array(mvp_mat.z),
        //     to_array(mvp_mat.w),
        // ];

        self.update_priorities(camera);
        // self.update_cache(encoder);
        self.update_visibility(cull_frustum);
        // self.update_shaders();

        // self.ocean.update(encoder, dt);

        // self.pipeline_data.model_view_projection = mvp_mat;
        // self.pipeline_data.camera_position = [camera.x, camera.y, camera.z];
        // self.pipeline_data.sun_direction = sun_direction;

        // let inv_mvp_mat = vecmath::mat4_inv::<f64>(vecmath::mat4_cast(mvp_mat));
        // let homogeneous = |[x, y, z, w]: [f64; 4]| [x / w, y / w, z / w];
        // let unproject = |v| homogeneous(vecmath::col_mat4_transform(inv_mvp_mat, v));
        // let ray = |x, y| {
        //     vecmath::vec3_cast(vecmath::vec3_normalized(vecmath::vec3_sub(
        //         unproject([x, y, 0.5, 1.0]),
        //         unproject([x, y, 1.0, 1.0]),
        //     )))
        // };
        // self.sky_pipeline_data.ray_bottom_left = ray(-1.0, -1.0);
        // self.sky_pipeline_data.ray_bottom_right = ray(1.0, -1.0);
        // self.sky_pipeline_data.ray_top_left = ray(-1.0, 1.0);
        // self.sky_pipeline_data.ray_top_right = ray(1.0, 1.0);
        // self.sky_pipeline_data.camera_position = [camera.x, camera.y, camera.z];
        // self.sky_pipeline_data.sun_direction = sun_direction;

        // self.planet_mesh_pipeline_data.model_view_projection = mvp_mat;
        // self.planet_mesh_pipeline_data.camera_position = [camera.x, camera.y, camera.z];
        // self.planet_mesh_pipeline_data.sun_direction = sun_direction;

        // self.instanced_mesh_pipeline_data.model_view_projection = mvp_mat;
        // self.instanced_mesh_pipeline_data.camera_position = [camera.x, camera.y, camera.z];
        // self.instanced_mesh_pipeline_data.sun_direction = sun_direction;
    }

    fn breadth_first<Visit>(&mut self, mut visit: Visit)
    where
        Visit: FnMut(&mut Self, NodeId) -> bool,
    {
        let mut pending = VecDeque::new();
        if visit(self, NodeId::root()) {
            pending.push_back(NodeId::root());
        }
        while let Some(id) = pending.pop_front() {
            for i in 0..4 {
                if let Some(child) = self.nodes[id].children[i] {
                    if visit(self, child) {
                        pending.push_back(child);
                    }
                }
            }
        }
    }

    pub fn get_height(&self, p: Point2<f32>) -> Option<f32> {
        if self.nodes[0].bounds.min.x > p.x
            || self.nodes[0].bounds.max.x < p.x
            || self.nodes[0].bounds.min.z > p.y
            || self.nodes[0].bounds.max.z < p.y
        {
            return None;
        }

        let mut id = NodeId::root();
        while self.nodes[id].children.iter().any(|c| c.is_some()) {
            for c in self.nodes[id].children.iter() {
                if let Some(c) = *c {
                    if self.nodes[c].bounds.min.x <= p.x
                        && self.nodes[c].bounds.max.x >= p.x
                        && self.nodes[c].bounds.min.z <= p.y
                        && self.nodes[c].bounds.max.z >= p.y
                    {
                        id = c;
                        break;
                    }
                }
            }
        }

        let layer = &self.tile_cache_layers[LayerType::Heights.index()];
        let resolution = (layer.resolution() - 1) as f32;
        let x = (p.x - self.nodes[id].bounds.min.x) / self.nodes[id].side_length * resolution;
        let y = (p.y - self.nodes[id].bounds.min.z) / self.nodes[id].side_length * resolution;

        let get_texel = |x, y| {
            layer
                .get_texel(&self.nodes[id], x, y)
                .read_f32::<LittleEndian>()
                .unwrap()
        };

        let (mut fx, mut fy) = (x.fract(), y.fract());
        let (mut ix, mut iy) = (x.floor() as usize, y.floor() as usize);
        if ix == layer.resolution() as usize - 1 {
            ix = layer.resolution() as usize - 2;
            fx = 1.0;
        }
        if iy == layer.resolution() as usize - 1 {
            iy = layer.resolution() as usize - 2;
            fy = 1.0;
        }

        if fx + fy < 1.0 {
            Some(
                (1.0 - fx - fy) * get_texel(ix, iy)
                    + fx * get_texel(ix + 1, iy)
                    + fy * get_texel(ix, iy + 1),
            )
        } else {
            Some(
                (fx + fy - 1.0) * get_texel(ix + 1, iy + 1)
                    + (1.0 - fx) * get_texel(ix, iy + 1)
                    + (1.0 - fy) * get_texel(ix + 1, iy),
            )
        }
    }

    // fn make_index_buffers(&mut self, factory: &Factory<B>, queue: QueueId, index: usize, subpass: &Subpass<B>) {
    //     let (index_buffer, index_buffer_partial) = {
    //         let mut make_index_buffer = |resolution: u32| -> Result<Escape<Buffer<B>>, Error> {
    //             fn make_indices_inner<B: Backend, T>(
    //                 factory: &mut Factory<B>,
    //                 resolution: u32,
    //             ) -> Result<Escape<Buffer<B>>, Error>
    //             where
    //                 T: TryFrom<u32>,
    //                 <T as TryFrom<u32>>::Error: Debug,
    //             {
    //                 let width = resolution + 1;
    //                 let mut indices = Vec::new();
    //                 for y in 0..resolution {
    //                     for x in 0..resolution {
    //                         for offset in [0, 1, width, 1, width + 1, width].iter() {
    //                             indices.push(T::try_from(offset + (x + y * width)).unwrap());
    //                         }
    //                     }
    //                 }
    //                 let buffer = factory.create_buffer(
    //                     rendy::resource::BufferInfo {
    //                         size: (indices.len() * mem::size_of::<T>()) as u64,
    //                         usage: gfx_hal::buffer::Usage::INDEX,
    //                     },
    //                     rendy::memory::Data,
    //                 )?;
    //                 unsafe {
    //                     factory.upload_buffer(&buffer, 0, &indices, None, unimplemented!())?;
    //                 }
    //                 Ok(buffer)
    //             }

    //             if (resolution + 1) * (resolution + 1) - 1 <= u16::max_value() as u32 {
    //                 make_indices_inner::<B, u16>(factory, resolution)
    //             } else {
    //                 make_indices_inner::<B, u32>(factory, resolution)
    //             }
    //         };
    //         let resolution =
    //             (tile_cache_layers[LayerType::Heights.index()].resolution() - 1) as u32;
    //         (
    //             make_index_buffer(resolution)?,
    //             make_index_buffer(resolution / 2)?,
    //         )
    //     };
    // }

    pub(crate) fn total_nodes(&self) -> usize {
        self.nodes.len()
    }
}
