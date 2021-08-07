use crate::cache::Priority;
use crate::cache::TileCache;
use crate::utils::math::InfiniteFrustum;
use cgmath::*;
use fnv::FnvHashMap;
use wgpu::util::DeviceExt;

pub(crate) mod node;
pub(crate) mod render;

pub(crate) use crate::terrain::quadtree::node::*;

/// The central object in terra. It holds all relevant state and provides functions to update and
/// render the terrain.
pub(crate) struct QuadTree {
    visible_nodes: Vec<VNode>,
    partially_visible_nodes: Vec<(VNode, u8)>,

    heights_resolution: u32,

    node_priorities: FnvHashMap<VNode, Priority>,
    last_camera_position: Option<mint::Point3<f64>>,
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
            heights_resolution,
            node_priorities: FnvHashMap::default(),
            last_camera_position: None,
        }
    }

    pub(crate) fn create_index_buffers(&self, device: &wgpu::Device) -> wgpu::Buffer {
        let mut data = Vec::new();
        let resolution = self.heights_resolution as u16;
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

        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            usage: wgpu::BufferUsage::INDEX,
            label: Some("buffer.terrain.index"),
            contents: bytemuck::cast_slice(&data),
        })
    }

    pub fn update_priorities(&mut self, tile_cache: &TileCache, camera: mint::Point3<f64>) {
        if self.last_camera_position == Some(camera) {
            return;
        }
        self.last_camera_position = Some(camera);
        let camera = Vector3::new(camera.x, camera.y, camera.z);

        self.node_priorities.clear();
        VNode::breadth_first(|node| {
            let priority = node.priority(camera, tile_cache.get_height_range(node));
            self.node_priorities.insert(node, priority);
            priority >= Priority::cutoff() && node.level() < VNode::LEVEL_CELL_5MM
        });
    }

    pub fn update_visibility(
        &mut self,
        tile_cache: &TileCache,
        frustum: &InfiniteFrustum,
        camera: mint::Point3<f64>,
    ) {
        self.visible_nodes.clear();
        self.partially_visible_nodes.clear();

        // Any node with all needed layers in cache is visible...
        let mut node_visibilities: FnvHashMap<VNode, bool> = FnvHashMap::default();
        VNode::breadth_first(|node| {
            let priority = self.node_priority(node);
            let visible = (node.level() == 0 || priority >= Priority::cutoff())
                && node.in_frustum(&frustum, tile_cache.get_height_range(node));

            node_visibilities.insert(node, visible);
            visible && node.level() < VNode::LEVEL_CELL_5MM
        });
        // let min_missing_level = node_visibilities
        //     .iter()
        //     .filter(|(&n, &v)| v && !tile_cache.contains(n, LayerType::Displacements))
        //     .map(|(n, v)| n.level())
        //     .min();
        // if let Some(min) = min_missing_level {
        //     for (n, v) in node_visibilities.iter_mut() {
        //         if n.level() >= min {
        //             *v = false;
        //         }
        //     }
        // }

        // ...Except if all its children are visible instead.
        VNode::breadth_first(|node| {
            if node.level() < VNode::LEVEL_CELL_5MM && node_visibilities[&node] {
                let mut mask = 0;
                for (i, c) in node.children().iter().enumerate() {
                    if !node_visibilities[c] {
                        mask = mask | (1 << i);
                    }
                }

                if mask == 15 {
                    self.visible_nodes.push(node);
                } else if mask > 0 {
                    self.partially_visible_nodes.push((node, mask));
                }

                mask < 15
            } else if node_visibilities[&node] {
                self.visible_nodes.push(node);
                false
            } else {
                false
            }
        });
    }

    pub fn node_priority(&self, node: VNode) -> Priority {
        self.node_priorities.get(&node).cloned().unwrap_or(Priority::none())
    }
}
