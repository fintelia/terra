use cgmath::*;

use std::collections::VecDeque;
use std::ops::{Index, IndexMut};

use terrain::tile_cache::{Priority, TileCache};
use utils::math::BoundingBox;

#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
pub struct NodeId(u32);
impl NodeId {
    pub fn root() -> Self {
        NodeId(0)
    }
    pub fn index(&self) -> usize {
        self.0 as usize
    }
}
impl Index<NodeId> for Vec<Node> {
    type Output = Node;
    fn index(&self, id: NodeId) -> &Node {
        &self[id.0 as usize]
    }
}
impl IndexMut<NodeId> for Vec<Node> {
    fn index_mut(&mut self, id: NodeId) -> &mut Node {
        &mut self[id.0 as usize]
    }
}

const NUM_LAYERS: usize = 3;

pub struct Node {
    level: u8,
    #[allow(unused)]
    parent: Option<NodeId>,
    children: [Option<NodeId>; 4],

    bounds: BoundingBox,
    min_distance: f32,

    size: i32,
    center: Point2<i32>,

    /// Index of this node in the tile list.
    tile_indices: [Option<u32>; NUM_LAYERS],

    /// How much this node is needed for the current frame. Nodes with priority less than 1.0 will
    /// not be rendered (they are too detailed).
    priority: Priority,

    visible: bool,
}
impl Node {
    pub fn priority(&self) -> Priority {
        self.priority
    }
}


pub struct QuadTree {
    /// List of nodes in the `QuadTree`. The root is always at index 0.
    nodes: Vec<Node>,

    /// List of nodes that will be rendered.
    visible_nodes: Vec<NodeId>,

    /// Cache holding nearby tiles for each layer.
    tile_cache_layers: [TileCache; NUM_LAYERS],
}

#[allow(unused)]
impl QuadTree {
    pub fn new(side_length: f32, playable_radius: f32, max_level: u8) -> Self {
        let node = Node {
            level: 0,
            parent: None,
            children: [None; 4],
            bounds: BoundingBox {
                min: Point3::new(-side_length * 0.5, 0.0, -side_length * 0.5),
                max: Point3::new(side_length * 0.5, 16000.0, side_length * 0.5),
            },
            min_distance: side_length * 4.,
            center: Point2::origin(),
            size: 1 << 31,
            priority: Priority::none(),
            tile_indices: [None; NUM_LAYERS],
            visible: false,
        };

        let mut nodes = vec![node];
        let mut pending = VecDeque::new();
        pending.push_back(NodeId::root());

        while let Some(parent) = pending.pop_front() {
            if nodes[parent].level >= max_level {
                continue;
            }

            let min_distance = nodes[parent].min_distance * 0.5;
            let min = nodes[parent].bounds.min;
            let max = nodes[parent].bounds.max;
            let center = Point3::midpoint(min, max);
            let bounds = [
                BoundingBox::new(
                    Point3::new(min.x, min.y, min.z),
                    Point3::new(center.x, max.y, center.z),
                ),
                BoundingBox::new(
                    Point3::new(center.x, min.y, min.z),
                    Point3::new(max.x, max.y, center.z),
                ),
                BoundingBox::new(
                    Point3::new(min.x, min.y, center.z),
                    Point3::new(center.x, max.y, max.z),
                ),
                BoundingBox::new(
                    Point3::new(center.x, min.y, center.z),
                    Point3::new(max.x, max.y, max.z),
                ),
            ];

            let offsets = [
                Vector2::new(-1, -1),
                Vector2::new(1, -1),
                Vector2::new(-1, 1),
                Vector2::new(1, 1),
            ];

            for i in 0..4 {
                if bounds[i].distance(Point3::origin()) < playable_radius + min_distance {
                    let child = NodeId(nodes.len() as u32);
                    let child_node = Node {
                        level: nodes[parent].level + 1,
                        parent: Some(parent),
                        children: [None; 4],
                        bounds: bounds[i],
                        min_distance: nodes[parent].min_distance * 0.5,
                        size: nodes[parent].size / 2,
                        center: Point2::from_vec(
                            nodes[parent].center.to_vec() - offsets[i] * nodes[parent].size / 4,
                        ),
                        priority: Priority::none(),
                        tile_indices: [None; NUM_LAYERS],
                        visible: false,
                    };

                    nodes.push(child_node);
                    nodes[parent].children[i] = Some(child);
                    pending.push_back(child);
                }
            }
        }

        Self {
            nodes,
            visible_nodes: Vec::new(),
            tile_cache_layers: [
                TileCache::new(1024),
                TileCache::new(512),
                TileCache::new(96),
            ],
        }
    }

    fn update_priorities(&mut self, camera: Point3<f32>) {
        for node in self.nodes.iter_mut() {
            node.priority = Priority::from_f32(
                node.bounds.square_distance(camera).max(1.0) /
                    (node.min_distance * node.min_distance),
            );
        }
        for cache_layer in self.tile_cache_layers.iter_mut() {
            cache_layer.update_priorities(&mut self.nodes);
        }
    }

    pub fn update(&mut self, camera: Point3<f32>) {
        self.update_priorities(camera);

        // Update tile cache.
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
        for cache_layer in self.tile_cache_layers.iter_mut() {
            cache_layer.load_missing(&mut self.nodes);
        }

        // Determine visible nodes.
        self.visible_nodes.clear();
        for node in self.nodes.iter_mut() {
            node.visible = false;
        }
        // Any node with all needed layers in cache is visible...
        self.breadth_first(|qt, id| {
            qt.nodes[id].visible = qt.nodes[id].tile_indices.iter().filter_map(|i| *i).all(
                |i| {
                    qt.tile_cache_layers[i as usize].contains(id)
                },
            );
            qt.nodes[id].visible
        });
        // ...Except if all its children are visible instead.
        self.breadth_first(|qt, id| if qt.nodes[id].visible {
            qt.nodes[id].visible = !qt.nodes[id].children.iter().all(|child| match *child {
                Some(c) => qt.nodes[c].visible,
                None => false,
            });
            if qt.nodes[id].visible {
                qt.visible_nodes.push(id);
            }
            true
        } else {
            false
        });
    }

    fn breadth_first<F>(&mut self, mut visit: F)
    where
        F: FnMut(&mut Self, NodeId) -> bool,
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
