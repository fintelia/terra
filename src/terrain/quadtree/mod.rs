use byteorder::{LittleEndian, ReadBytesExt};
use cgmath::*;
use collision::{Frustum, Relation};
use failure::Error;
use gfx;
use gfx::traits::FactoryExt;
use gfx_core;
use memmap::Mmap;
use std::convert::TryFrom;
use vecmath;
use vec_map::VecMap;

use rshader;

use std::collections::VecDeque;
use std::fmt::Debug;
use std::env;

use cache::AssetLoadContext;
use terrain::material::MaterialSet;
use terrain::tile_cache::{LayerType, Priority, TileCache, TileHeader, NUM_LAYERS};

use sky::{Skybox, Atmosphere};
use ocean::Ocean;

pub(crate) mod id;
pub(crate) mod node;
pub(crate) mod render;

pub(crate) use terrain::quadtree::id::*;
pub(crate) use terrain::quadtree::node::*;
pub(crate) use terrain::quadtree::render::*;

/// The central object in terra. It holds all relevant state and provides functions to update and
/// render the terrain.
pub struct QuadTree<R, F>
where
    R: gfx::Resources,
    F: gfx::Factory<R>,
{
    /// List of nodes in the `QuadTree`. The root is always at index 0.
    nodes: Vec<Node>,
    ocean: Ocean<R>,
    atmosphere: Atmosphere<R>,

    /// List of nodes that will be rendered.
    visible_nodes: Vec<NodeId>,
    partially_visible_nodes: Vec<(NodeId, u8)>,

    /// Cache holding nearby tiles for each layer.
    tile_cache_layers: VecMap<TileCache<R>>,

    index_buffer: gfx::IndexBuffer<R>,
    index_buffer_partial: gfx::IndexBuffer<R>,

    factory: F,
    pso: gfx::PipelineState<R, pipe::Meta>,
    pipeline_data: pipe::Data<R>,
    sky_pso: gfx::PipelineState<R, sky_pipe::Meta>,
    sky_pipeline_data: sky_pipe::Data<R>,
    planet_mesh_pso: gfx::PipelineState<R, planet_mesh_pipe::Meta>,
    planet_mesh_pipeline_data: planet_mesh_pipe::Data<R>,
    shaders_watcher: rshader::ShaderDirectoryWatcher,
    shader: rshader::Shader<R>,
    sky_shader: rshader::Shader<R>,
    planet_mesh_shader: rshader::Shader<R>,
    num_planet_mesh_vertices: usize,
    node_states: Vec<NodeState>,
    _materials: MaterialSet<R>,
}

#[allow(unused)]
impl<R, F> QuadTree<R, F>
where
    R: gfx::Resources,
    F: gfx::Factory<R>,
{
    pub(crate) fn new(
        header: TileHeader,
        data_file: Mmap,
        materials: MaterialSet<R>,
        sky: Skybox<R>,
        mut factory: F,
        color_buffer: &gfx::handle::RenderTargetView<R, gfx::format::Srgba8>,
        depth_buffer: &gfx::handle::DepthStencilView<R, gfx::format::DepthStencil>,
    ) -> Result<Self, Error> {
        let mut shaders_watcher =
            rshader::ShaderDirectoryWatcher::new(env::var("TERRA_SHADER_DIRECTORY").unwrap_or(
                "src/shaders/glsl".to_string(),
            ))?;

        let shader = rshader::Shader::simple(
            &mut factory,
            &mut shaders_watcher,
            shader_source!(
                "../../shaders/glsl",
                "version",
                "atmosphere",
                "terrain.glslv"
            ),
            shader_source!(
                "../../shaders/glsl",
                "version",
                "atmosphere",
                "hash",
                "terrain.glslf"
            ),
        )?;

        let sky_shader = rshader::Shader::simple(
            &mut factory,
            &mut shaders_watcher,
            shader_source!("../../shaders/glsl", "version", "sky.glslv"),
            shader_source!(
                "../../shaders/glsl",
                "version",
                "atmosphere",
                "hash",
                "sky.glslf"
            ),
        )?;

        let planet_mesh_shader =
            rshader::Shader::simple(
                &mut factory,
                &mut shaders_watcher,
                shader_source!("../../shaders/glsl", "version", "planet_mesh.glslv"),
                shader_source!(
                    "../../shaders/glsl",
                    "version",
                    "atmosphere",
                    "hash",
                    "planet_mesh.glslf"
                ),
            )?;

        let mut data_view = data_file.into_view_sync();
        let mut tile_cache_layers = VecMap::new();
        for layer in header.layers.iter().cloned() {
            tile_cache_layers.insert(
                layer.layer_type as usize,
                TileCache::new(layer, unsafe { data_view.clone() }, &mut factory),
            );
        }

        let noise_start = header.noise.offset;
        let noise_end = noise_start + header.noise.bytes;
        let noise_data = unsafe { &data_view.as_slice()[noise_start..noise_end] };
        let (noise_texture, noise_texture_view) =
            factory.create_texture_immutable_u8
            ::<(gfx_core::format::R8_G8_B8_A8, gfx_core::format::Unorm)>(
                gfx::texture::Kind::D2(
                    header.noise.resolution as u16,
                    header.noise.resolution as u16,
                    gfx::texture::AaMode::Single,
                ),
                gfx::texture::Mipmap::Provided,
                &[gfx::memory::cast_slice(noise_data)],
            )?;

        let planet_mesh_start = header.planet_mesh.offset;
        let planet_mesh_end = planet_mesh_start + header.planet_mesh.bytes;
        let planet_mesh_data = unsafe { &data_view.as_slice()[planet_mesh_start..planet_mesh_end] };
        let planet_mesh_vertices = gfx::memory::cast_slice(planet_mesh_data);

        let pm_texture_start = header.planet_mesh_texture.offset;
        let pm_texture_end = pm_texture_start + header.planet_mesh_texture.bytes;
        let pm_texture_data = unsafe { &data_view.as_slice()[pm_texture_start..pm_texture_end] };
        let (planet_mesh_texture, planet_mesh_texture_view) =
            factory.create_texture_immutable_u8
            ::<(gfx_core::format::R8_G8_B8_A8, gfx_core::format::Srgb)>(
                gfx::texture::Kind::D2(
                    header.planet_mesh_texture.resolution as u16,
                    header.planet_mesh_texture.resolution as u16,
                    gfx::texture::AaMode::Single,
                ),
                gfx::texture::Mipmap::Provided,
                &[gfx::memory::cast_slice(pm_texture_data)],
            )?;

        let heights_texture_view = tile_cache_layers[LayerType::Heights.index()]
            .get_texture_view_f32()
            .unwrap()
            .clone();
        let colors_texture_view = tile_cache_layers[LayerType::Colors.index()]
            .get_texture_view_srgba()
            .unwrap()
            .clone();
        let normals_texture_view = tile_cache_layers[LayerType::Normals.index()]
            .get_texture_view_rgba8()
            .unwrap()
            .clone();
        let water_texture_view = tile_cache_layers[LayerType::Water.index()]
            .get_texture_view_rgba8()
            .unwrap()
            .clone();

        let ocean = Ocean::new(&mut factory);
        let atmosphere = Atmosphere::new(&mut factory, &mut AssetLoadContext::new())?;

        let sampler = factory.create_sampler(gfx::texture::SamplerInfo::new(
            gfx::texture::FilterMethod::Trilinear,
            gfx::texture::WrapMode::Clamp,
        ));

        let sampler_wrap = factory.create_sampler(gfx::texture::SamplerInfo::new(
            gfx::texture::FilterMethod::Trilinear,
            gfx::texture::WrapMode::Tile,
        ));

        // Extra scope to work around lack of non-lexical lifetimes.
        let (index_buffer, index_buffer_partial) = {
            let mut make_index_buffer = |resolution: u32| -> Result<gfx::IndexBuffer<R>, Error> {
                fn make_indices_inner<R: gfx::Resources, F: gfx::Factory<R>, T>(
                    factory: &mut F,
                    resolution: u32,
                ) -> Result<gfx::handle::Buffer<R, T>, Error>
                where
                    T: TryFrom<u32> + gfx_core::memory::Pod,
                    <T as TryFrom<u32>>::Error: Debug,
                {
                    let width = resolution + 1;
                    let mut indices = Vec::new();
                    for y in 0..resolution {
                        for x in 0..resolution {
                            for offset in [0, 1, width, 1, width + 1, width].iter() {
                                indices.push(T::try_from(offset + (x + y * width)).unwrap());
                            }
                        }
                    }
                    Ok(factory.create_buffer_immutable(
                        &indices[..],
                        gfx::buffer::Role::Index,
                        gfx::memory::Bind::empty(),
                    )?)
                }
                Ok(if (resolution + 1) * (resolution + 1) - 1 <=
                    u16::max_value() as u32
                {
                    gfx::IndexBuffer::Index16(make_indices_inner(&mut factory, resolution)?)
                } else {
                    gfx::IndexBuffer::Index32(make_indices_inner(&mut factory, resolution)?)
                })
            };
            let resolution = (tile_cache_layers[LayerType::Heights.index()].resolution() - 1) as
                u32;
            (
                make_index_buffer(resolution)?,
                make_index_buffer(resolution / 2)?,
            )
        };

        let transmittance = (
            atmosphere.transmittance.texture_view.clone(),
            sampler.clone(),
        );
        let inscattering = (
            atmosphere.inscattering.texture_view.clone(),
            sampler.clone(),
        );

        Ok(Self {
            visible_nodes: Vec::new(),
            partially_visible_nodes: Vec::new(),
            index_buffer,
            index_buffer_partial,
            pso: Self::make_pso(&mut factory, shader.as_shader_set())?,
            pipeline_data: pipe::Data {
                instances: factory.create_constant_buffer::<NodeState>(header.nodes.len() * 3),
                model_view_projection: [[0.0; 4]; 4],
                camera_position: [0.0, 0.0, 0.0],
                sun_direction: [0.0, 0.70710678118, 0.70710678118],
                resolution: 0,
                heights: (heights_texture_view, sampler.clone()),
                colors: (colors_texture_view, sampler.clone()),
                normals: (normals_texture_view, sampler.clone()),
                water: (water_texture_view, sampler.clone()),
                materials: (materials.texture_view.clone(), sampler_wrap.clone()),
                sky: (sky.texture_view.clone(), sampler.clone()),
                ocean_surface: (ocean.texture_view.clone(), sampler_wrap.clone()),
                noise: (noise_texture_view, sampler_wrap),
                noise_wavelength: header.noise.wavelength,
                planet_radius: 6371000.0,
                atmosphere_radius: 6471000.0,
                transmittance: transmittance.clone(),
                inscattering: inscattering.clone(),
                color_buffer: color_buffer.clone(),
                depth_buffer: depth_buffer.clone(),
            },
            sky_pso: Self::make_sky_pso(&mut factory, sky_shader.as_shader_set())?,
            sky_pipeline_data: sky_pipe::Data {
                inv_model_view_projection: [[0.0; 4]; 4],
                camera_position: [0.0, 0.0, 0.0],
                sun_direction: [0.0, 0.70710678118, 0.70710678118],
                sky: (sky.texture_view.clone(), sampler.clone()),
                planet_radius: 6371000.0,
                atmosphere_radius: 6471000.0,
                transmittance: transmittance.clone(),
                inscattering: inscattering.clone(),
                color_buffer: color_buffer.clone(),
                depth_buffer: depth_buffer.clone(),
            },
            planet_mesh_pso: Self::make_planet_mesh_pso(
                &mut factory,
                planet_mesh_shader.as_shader_set(),
            )?,
            planet_mesh_pipeline_data: planet_mesh_pipe::Data {
                vertices: factory.create_vertex_buffer(planet_mesh_vertices),
                model_view_projection: [[0.0; 4]; 4],
                camera_position: [0.0, 0.0, 0.0],
                sun_direction: [0.0, 0.70710678118, 0.70710678118],
                planet_radius: 6371000.0,
                atmosphere_radius: 6471000.0,
                transmittance: transmittance,
                inscattering: inscattering,
                color: (planet_mesh_texture_view, sampler.clone()),
                color_buffer: color_buffer.clone(),
                depth_buffer: depth_buffer.clone(),
            },
            ocean,
            atmosphere,
            factory,
            shaders_watcher,
            shader,
            sky_shader,
            planet_mesh_shader,
            num_planet_mesh_vertices: header.planet_mesh.num_vertices,
            nodes: header.nodes,
            node_states: Vec::new(),
            tile_cache_layers,
            _materials: materials,
        })
    }

    fn update_priorities(&mut self, camera: Point3<f32>) {
        for node in self.nodes.iter_mut() {
            node.priority = Priority::from_f32(
                (node.min_distance * node.min_distance) /
                    node.bounds.square_distance(camera).max(0.001),
            );
        }
        for (_, ref mut cache_layer) in self.tile_cache_layers.iter_mut() {
            cache_layer.update_priorities(&mut self.nodes);
        }
    }

    fn update_cache<C: gfx_core::command::Buffer<R>>(&mut self, encoder: &mut gfx::Encoder<R, C>) {
        self.breadth_first(|qt, id| {
            if qt.nodes[id].priority < Priority::cutoff() {
                return false;
            }

            for layer in 0..NUM_LAYERS {
                if qt.nodes[id].tile_indices[layer].is_some() &&
                    !qt.tile_cache_layers[layer].contains(id)
                {
                    qt.tile_cache_layers[layer].add_missing((qt.nodes[id].priority, id));
                }
            }
            true
        });
        for (_, ref mut cache_layer) in self.tile_cache_layers.iter_mut() {
            cache_layer.load_missing(&mut self.nodes, encoder);
        }
    }

    fn update_visibility(&mut self, cull_frustum: Option<Frustum<f32>>) {
        self.visible_nodes.clear();
        self.partially_visible_nodes.clear();
        for node in self.nodes.iter_mut() {
            node.visible = false;
        }
        // Any node with all needed layers in cache is visible...
        self.breadth_first(|qt, id| {
            qt.nodes[id].visible = id == NodeId::root() ||
                qt.nodes[id].priority >= Priority::cutoff();
            qt.nodes[id].visible
        });
        // ...Except if all its children are visible instead.
        self.breadth_first(|qt, id| if qt.nodes[id].visible {
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
        });
    }

    pub fn update<C: gfx_core::command::Buffer<R>>(
        &mut self,
        mvp_mat: Matrix4<f32>,
        camera: Point3<f32>,
        cull_frustum: Option<Frustum<f32>>,
        encoder: &mut gfx::Encoder<R, C>,
        dt: f32,
    ) {
        // Convert the MVP matrix to "vecmath encoding".
        let to_array = |v: Vector4<f32>| [v.x, v.y, v.z, v.w];
        let mvp_mat = [
            to_array(mvp_mat.x),
            to_array(mvp_mat.y),
            to_array(mvp_mat.z),
            to_array(mvp_mat.w),
        ];

        self.update_priorities(camera);
        self.update_cache(encoder);
        self.update_visibility(cull_frustum);
        self.update_shaders();

        self.ocean.update(encoder, dt);

        self.pipeline_data.model_view_projection = mvp_mat;
        self.pipeline_data.camera_position = [camera.x, camera.y, camera.z];

        self.sky_pipeline_data.inv_model_view_projection = vecmath::mat4_inv(mvp_mat);
        self.sky_pipeline_data.camera_position = [camera.x, camera.y, camera.z];

        self.planet_mesh_pipeline_data.model_view_projection = mvp_mat;
        self.planet_mesh_pipeline_data.camera_position = [camera.x, camera.y, camera.z];
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
        if self.nodes[0].bounds.min.x > p.x || self.nodes[0].bounds.max.x < p.x ||
            self.nodes[0].bounds.min.z > p.y || self.nodes[0].bounds.max.z < p.y
        {
            return None;
        }

        let mut id = NodeId::root();
        while self.nodes[id].children.iter().any(|c| c.is_some()) {
            for c in self.nodes[id].children.iter() {
                if let Some(c) = *c {
                    if self.nodes[c].bounds.min.x <= p.x && self.nodes[c].bounds.max.x >= p.x &&
                        self.nodes[c].bounds.min.z <= p.y &&
                        self.nodes[c].bounds.max.z >= p.y
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
                (1.0 - fx - fy) * get_texel(ix, iy) + fx * get_texel(ix + 1, iy) +
                    fy * get_texel(ix, iy + 1),
            )
        } else {
            Some(
                (fx + fy - 1.0) * get_texel(ix + 1, iy + 1) + (1.0 - fx) * get_texel(ix, iy + 1) +
                    (1.0 - fy) * get_texel(ix + 1, iy),
            )
        }
    }
}
