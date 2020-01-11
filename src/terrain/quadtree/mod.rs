use crate::terrain::tile_cache::{Priority, TileCache};
use byteorder::{ByteOrder, NativeEndian};
use cgmath::*;
use collision::{Frustum, Relation};
use std::collections::{HashMap};
use vec_map::VecMap;

pub(crate) mod node;
pub(crate) mod render;

pub(crate) use crate::terrain::quadtree::node::*;
pub(crate) use crate::terrain::quadtree::render::*;

/// The central object in terra. It holds all relevant state and provides functions to update and
/// render the terrain.
pub(crate) struct QuadTree {
    // ocean: Ocean<R>,
    /// List of nodes that will be rendered.
    visible_nodes: Vec<VNode>,
    partially_visible_nodes: Vec<(VNode, u8)>,

    heights_resolution: u32,

    node_states: Vec<NodeState>,
}

impl std::fmt::Debug for QuadTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QuadTree")
    }
}

#[allow(unused)]
impl QuadTree {
    pub(crate) fn new(heights_resolution: u32) -> Self {
        Self {
            visible_nodes: Vec::new(),
            partially_visible_nodes: Vec::new(),
            node_states: Vec::new(),
            heights_resolution,
        }
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
                    for offset in [0, 1, width, 1, width + 1, width].iter() {
                        NativeEndian::write_u16(&mut mapped.data[i..], offset + (x + y * width));
                        i += 2;
                    }
                }
            }

            mapped.finish()
        };
        let resolution = self.heights_resolution as u16;

        (make_index_buffer(resolution), make_index_buffer(resolution / 2))
    }

    fn update_cache(&mut self, tile_cache: &mut VecMap<TileCache>, camera: Point3<f32>) {
        for (_, ref mut cache_layer) in tile_cache.iter_mut() {
            cache_layer.update_priorities(camera);
        }

        VNode::breadth_first(|node| {
            let priority = node.priority(camera);
            if priority < Priority::cutoff() {
                return false;
            }

            for layer in tile_cache.values_mut() {
                layer.add_missing((priority, node));
            }
            true
        });
    }

    fn update_visibility(&mut self, camera: Point3<f32>, cull_frustum: Option<Frustum<f32>>) {
        self.visible_nodes.clear();
        self.partially_visible_nodes.clear();

        let mut node_visibilities: HashMap<VNode, bool> = HashMap::new();

        // Any node with all needed layers in cache is visible...
        VNode::breadth_first(|node| {
            let visible = node == VNode::root() || node.priority(camera) >= Priority::cutoff();
            node_visibilities.insert(node, visible);
            visible
        });
        // ...Except if all its children are visible instead.
        VNode::breadth_first(|node| {
            if node_visibilities[&node] {
                let mut mask = 0;
                let mut has_visible_children = false;
                for (i, c) in node.children().iter().enumerate() {
                    if !node_visibilities[c] {
                        mask = mask | (1 << i);
                    }

                    if node_visibilities[&c] {
                        has_visible_children = true;
                    }
                }

                if let Some(frustum) = cull_frustum {
                    // TODO: Also try to cull parts of a node, if contains() returns Relation::Cross.
                    if frustum.contains(&node.bounds().as_aabb3()) == Relation::Out {
                        mask = 0;
                    }
                }

                if mask == 0 {
                    node_visibilities.insert(node, false);
                } else if has_visible_children {
                    self.partially_visible_nodes.push((node, mask));
                } else {
                    self.visible_nodes.push(node);
                }

                true
            } else {
                false
            }
        });
    }

    pub fn update(
        &mut self,
        tile_cache: &mut VecMap<TileCache>,
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
        self.update_cache(tile_cache, camera);
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

    // fn breadth_first<Visit>(&mut self, mut visit: Visit)
    // where
    //     Visit: FnMut(&mut Self, NodeId) -> bool,
    // {
    //     let mut pending = VecDeque::new();
    //     if visit(self, NodeId::root()) {
    //         pending.push_back(NodeId::root());
    //     }
    //     while let Some(id) = pending.pop_front() {
    //         for i in 0..4 {
    //             if let Some(child) = self.nodes[id].children[i] {
    //                 if visit(self, child) {
    //                     pending.push_back(child);
    //                 }
    //             }
    //         }
    //     }
    // }

    // pub fn get_height(
    //     &self,
    //     mapfile: &MapFile,
    //     tile_cache: &VecMap<TileCache>,
    //     p: Point2<f32>,
    // ) -> Option<f32> {
    //     if self.nodes[0].bounds.min.x > p.x
    //         || self.nodes[0].bounds.max.x < p.x
    //         || self.nodes[0].bounds.min.z > p.y
    //         || self.nodes[0].bounds.max.z < p.y
    //     {
    //         return None;
    //     }

    //     let mut id = NodeId::root();
    //     while self.nodes[id].children.iter().any(|c| c.is_some()) {
    //         for c in self.nodes[id].children.iter() {
    //             if let Some(c) = *c {
    //                 if self.nodes[c].bounds.min.x <= p.x
    //                     && self.nodes[c].bounds.max.x >= p.x
    //                     && self.nodes[c].bounds.min.z <= p.y
    //                     && self.nodes[c].bounds.max.z >= p.y
    //                 {
    //                     id = c;
    //                     break;
    //                 }
    //             }
    //         }
    //     }

    //     let layer = &tile_cache[LayerType::Heights.index()];
    //     let resolution = (layer.resolution() - 1) as f32;
    //     let x = (p.x - self.nodes[id].bounds.min.x) / self.nodes[id].side_length() * resolution;
    //     let y = (p.y - self.nodes[id].bounds.min.z) / self.nodes[id].side_length() * resolution;

    //     let get_texel =
    //         |x, y| layer.get_texel(mapfile, id, x, y).read_f32::<LittleEndian>().unwrap();

    //     let (mut fx, mut fy) = (x.fract(), y.fract());
    //     let (mut ix, mut iy) = (x.floor() as usize, y.floor() as usize);
    //     if ix == layer.resolution() as usize - 1 {
    //         ix = layer.resolution() as usize - 2;
    //         fx = 1.0;
    //     }
    //     if iy == layer.resolution() as usize - 1 {
    //         iy = layer.resolution() as usize - 2;
    //         fy = 1.0;
    //     }

    //     if fx + fy < 1.0 {
    //         Some(
    //             (1.0 - fx - fy) * get_texel(ix, iy)
    //                 + fx * get_texel(ix + 1, iy)
    //                 + fy * get_texel(ix, iy + 1),
    //         )
    //     } else {
    //         Some(
    //             (fx + fy - 1.0) * get_texel(ix + 1, iy + 1)
    //                 + (1.0 - fx) * get_texel(ix, iy + 1)
    //                 + (1.0 - fy) * get_texel(ix + 1, iy),
    //         )
    //     }
    // }
}
