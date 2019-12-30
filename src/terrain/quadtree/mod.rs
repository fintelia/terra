use byteorder::{ByteOrder, LittleEndian, NativeEndian, ReadBytesExt};
use cgmath::*;
use collision::{Frustum, Relation};
use failure::Error;
use memmap::Mmap;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use vec_map::VecMap;

use crate::terrain::tile_cache::{
    LayerParams, LayerType, Priority, TileCache, TileHeader, NUM_LAYERS,
};

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
    pub(crate) tile_cache_layers: VecMap<TileCache>,

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
    // pub(crate) system: CoordinateSystem,
}

impl std::fmt::Debug for QuadTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QuadTree")
    }
}

#[allow(unused)]
impl QuadTree {
    pub(crate) fn new(
        data_file: Arc<Mmap>,
        cache_layers: Vec<LayerParams>,
        // materials: MaterialSet<R>,
        // sky: Skybox<R>,
        nodes: Vec<Node>,
    ) -> Result<Self, Error> {
        let mut tile_cache_layers = VecMap::new();
        for layer in cache_layers {
            tile_cache_layers
                .insert(layer.layer_type as usize, TileCache::new(layer, data_file.clone()));
        }

        Ok(Self {
            visible_nodes: Vec::new(),
            partially_visible_nodes: Vec::new(),
            nodes,
            node_states: Vec::new(),
            tile_cache_layers,
        })
    }

    pub(crate) fn create_index_buffers(
        &self,
        device: &wgpu::Device,
    ) -> (wgpu::Buffer, wgpu::Buffer) {
        let make_index_buffer = |resolution: u16| -> wgpu::Buffer {
            let mapped = device.create_buffer_mapped(
                12 * (resolution as usize + 1) * (resolution as usize + 1),
                wgpu::BufferUsage::INDEX,
            );

            let mut i = 0;
            let width = resolution + 1;
            for y in 0..resolution {
                for x in 0..resolution {
                    for offset in [0, 1, width, 1, width + 1, width].into_iter() {
                        NativeEndian::write_u16(&mut mapped.data[i..], offset + (x + y * width));
                        i += 2;
                    }
                }
            }

            mapped.finish()
        };
        let resolution =
            (self.tile_cache_layers[LayerType::Heights.index()].resolution() - 1) as u16;

        (make_index_buffer(resolution), make_index_buffer(resolution / 2))
    }

    fn update_cache(&mut self, camera: Point3<f32>) {
        for (_, ref mut cache_layer) in self.tile_cache_layers.iter_mut() {
            cache_layer.update_priorities(&mut self.nodes, camera);
        }

        self.breadth_first(|qt, id| {
            let priority = qt.nodes[id].priority(camera);
            if priority < Priority::cutoff() {
                return false;
            }

            for layer in 0..NUM_LAYERS {
                if qt.nodes[id].tile_indices[layer].is_some()
                    && !qt.tile_cache_layers[layer].contains(id)
                {
                    qt.tile_cache_layers[layer].add_missing((priority, id));
                }
            }
            true
        });
        for (_, ref mut cache_layer) in self.tile_cache_layers.iter_mut() {
            cache_layer.process_missing(&mut self.nodes);
        }
    }

    fn update_visibility(&mut self, camera: Point3<f32>, cull_frustum: Option<Frustum<f32>>) {
        self.visible_nodes.clear();
        self.partially_visible_nodes.clear();

        let mut node_visibilities: HashMap<NodeId, bool> = HashMap::new();

        // Any node with all needed layers in cache is visible...
        self.breadth_first(|qt, id| {
            let visible =
                id == NodeId::root() || qt.nodes[id].priority(camera) >= Priority::cutoff();
            node_visibilities.insert(id, visible);
            visible
        });
        // ...Except if all its children are visible instead.
        self.breadth_first(|qt, id| {
            if node_visibilities[&id] {
                let mut mask = 0;
                let mut has_visible_children = false;
                for (i, c) in qt.nodes[id].children.iter().enumerate() {
                    if c.is_none() || !node_visibilities[&c.unwrap()] {
                        mask = mask | (1 << i);
                    }

                    if c.is_some() && node_visibilities[&c.unwrap()] {
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
                    node_visibilities.insert(id, false);
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
        camera: mint::Point3<f32>,
        cull_frustum: Option<Frustum<f32>>,
        // dt: f32,
    ) {
        let camera = Point3::new(camera.x, camera.y, camera.z);

        // Convert the MVP matrix to "vecmath encoding".
        // let to_array = |v: Vector4<f32>| [v.x, v.y, v.z, v.w];
        // let mvp_mat = [
        //     to_array(mvp_mat.x),
        //     to_array(mvp_mat.y),
        //     to_array(mvp_mat.z),
        //     to_array(mvp_mat.w),
        // ];
        let start = std::time::Instant::now();
        self.update_cache(camera);
        self.update_visibility(camera, cull_frustum);

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
    }

    pub(crate) fn upload_tiles(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        state: &crate::GpuState,
    ) {
        self.tile_cache_layers[LayerType::Heights.index()].upload_tiles(
            device,
            encoder,
            &state.heights,
        );
        self.tile_cache_layers[LayerType::Normals.index()].upload_tiles(
            device,
            encoder,
            &state.normals,
        );
        self.tile_cache_layers[LayerType::Colors.index()].upload_tiles(
            device,
            encoder,
            &state.albedo,
        );
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

        let get_texel =
            |x, y| layer.get_texel(&self.nodes[id], x, y).read_f32::<LittleEndian>().unwrap();

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

    pub(crate) fn total_nodes(&self) -> usize {
        self.nodes.len()
    }
}
