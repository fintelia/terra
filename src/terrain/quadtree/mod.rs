use cgmath::*;
use gfx;
use gfx::traits::FactoryExt;
use gfx_core;
use memmap::Mmap;
use vecmath;
use vec_map::VecMap;

use rshader;

use std::env;
use std::collections::VecDeque;

use terrain::tile_cache::{LayerType, NUM_LAYERS, Priority, TileCache, TileHeader};

pub(crate) mod id;
pub(crate) mod node;
pub(crate) mod render;

pub(crate) use terrain::quadtree::id::*;
pub(crate) use terrain::quadtree::node::*;
pub(crate) use terrain::quadtree::render::*;

pub struct QuadTree<R, F>
where
    R: gfx::Resources,
    F: gfx::Factory<R>,
{
    /// List of nodes in the `QuadTree`. The root is always at index 0.
    nodes: Vec<Node>,

    /// List of nodes that will be rendered.
    visible_nodes: Vec<NodeId>,
    partially_visible_nodes: Vec<(NodeId, u8)>,

    /// Cache holding nearby tiles for each layer.
    tile_cache_layers: VecMap<TileCache<R>>,

    factory: F,
    pso: gfx::PipelineState<R, pipe::Meta>,
    pipeline_data: pipe::Data<R>,
    shaders_watcher: rshader::ShaderDirectoryWatcher,
    shader: rshader::Shader<R>,
    node_states: Vec<NodeState>,
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
        mut factory: F,
        color_buffer: &gfx::handle::RenderTargetView<R, gfx::format::Srgba8>,
        depth_buffer: &gfx::handle::DepthStencilView<R, gfx::format::DepthStencil>,
    ) -> Self {
        let mut shaders_watcher = rshader::ShaderDirectoryWatcher::new(
            env::var("TERRA_SHADER_DIRECTORY").unwrap_or(".".to_string()),
        ).unwrap();

        let shader = rshader::Shader::simple(
            &mut factory,
            &mut shaders_watcher,
            shader_source!("../../shaders/glsl", "version", "terrain.glslv"),
            shader_source!("../../shaders/glsl", "version", "terrain.glslf"),
        ).unwrap();

        let mut data_view = data_file.into_view_sync();
        let mut tile_cache_layers = VecMap::new();
        for layer in header.layers.iter().cloned() {
            tile_cache_layers.insert(
                layer.layer_type as usize,
                TileCache::new(layer, unsafe { data_view.clone() }, &mut factory),
            );
        }

        let heights_texture_view = tile_cache_layers[LayerType::Heights.index()]
            .get_texture_view_f32()
            .unwrap()
            .clone();
        let normals_texture_view = tile_cache_layers[LayerType::Normals.index()]
            .get_texture_view_rgba8()
            .unwrap()
            .clone();

        let sampler = factory.create_sampler(gfx::texture::SamplerInfo::new(
            gfx::texture::FilterMethod::Bilinear,
            gfx::texture::WrapMode::Clamp,
        ));

        Self {
            visible_nodes: Vec::new(),
            partially_visible_nodes: Vec::new(),
            pso: Self::make_pso(&mut factory, shader.as_shader_set()),
            pipeline_data: pipe::Data {
                instances: factory.create_constant_buffer::<NodeState>(header.nodes.len() * 3),
                model_view_projection: [[0.0; 4]; 4],
                camera_position: [0.0, 0.0, 0.0],
                resolution: 0,
                heights: (heights_texture_view, sampler.clone()),
                normals: (normals_texture_view, sampler),
                color_buffer: color_buffer.clone(),
                depth_buffer: depth_buffer.clone(),
            },
            factory,
            shaders_watcher,
            shader,
            nodes: header.nodes,
            node_states: Vec::new(),
            tile_cache_layers,
        }
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

    fn update_visibility(&mut self) {
        self.visible_nodes.clear();
        self.partially_visible_nodes.clear();
        for node in self.nodes.iter_mut() {
            node.visible = false;
        }
        // Any node with all needed layers in cache is visible...
        self.breadth_first(|qt, id| {
            qt.nodes[id].visible = qt.nodes[id].priority >= Priority::cutoff();
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
        mvp_mat: vecmath::Matrix4<f32>,
        camera: Point3<f32>,
        encoder: &mut gfx::Encoder<R, C>,
    ) {
        self.update_priorities(camera);
        self.update_cache(encoder);
        self.update_visibility();
        self.update_shaders();

        self.pipeline_data.model_view_projection = mvp_mat;
        self.pipeline_data.camera_position = [camera.x, camera.y, camera.z];
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
}