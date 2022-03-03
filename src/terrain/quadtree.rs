use crate::cache::TileCache;
use cgmath::*;
use fnv::FnvHashMap;
use types::{Priority, VNode, MAX_QUADTREE_LEVEL};
use wgpu::util::DeviceExt;

/// The central object in terra. It holds all relevant state and provides functions to update and
/// render the terrain.
pub(crate) struct QuadTree {
    node_priorities: FnvHashMap<VNode, Priority>,
    last_camera_position: Option<mint::Point3<f64>>,
}

impl QuadTree {
    pub(crate) fn new() -> Self {
        Self { node_priorities: FnvHashMap::default(), last_camera_position: None }
    }

    pub(crate) fn create_index_buffer(device: &wgpu::Device, resolution: u16) -> wgpu::Buffer {
        let mut data = Vec::new();
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
            usage: wgpu::BufferUsages::INDEX,
            label: Some("buffer.terrain.index"),
            contents: bytemuck::cast_slice(&data),
        })
    }

    pub fn update_priorities(&mut self, cache: &TileCache, camera: mint::Point3<f64>) {
        if self.last_camera_position == Some(camera) {
            return;
        }
        self.last_camera_position = Some(camera);
        let camera = Vector3::new(camera.x, camera.y, camera.z);

        self.node_priorities.clear();
        VNode::breadth_first(|node| {
            let priority = node.priority(camera, cache.get_height_range(node));
            self.node_priorities.insert(node, priority);
            priority >= Priority::cutoff() && node.level() < MAX_QUADTREE_LEVEL
        });
    }

    pub fn node_priority(&self, node: VNode) -> Priority {
        self.node_priorities.get(&node).cloned().unwrap_or(Priority::none())
    }
}
