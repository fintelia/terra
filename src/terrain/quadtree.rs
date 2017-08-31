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

pub enum NodeLayerIndices<T> {
    #[allow(unused)]
    Height(T),
    HeightNormal(T, T),
    #[allow(unused)]
    HeightNormalSplat(T, T, T),
}

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
    tile_indices: NodeLayerIndices<u32>,

    /// How much this node is needed for the current frame. Nodes with priority less than 1.0 will
    /// not be rendered (they are too detailed).
    priority: Priority,

    // Index of this node in each tile cache, if it is present.
    cache_indices: Option<NodeLayerIndices<u16>>,
}
impl Node {
    pub fn priority(&self) -> Priority {
        self.priority
    }
}

pub struct QuadTree {
    /// List of nodes in the `QuadTree`. The root is always at index 0.
    nodes: Vec<Node>,

    heights: TileCache,
    normals: TileCache,
    splats: TileCache,
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
            tile_indices: NodeLayerIndices::HeightNormal(0, 0),
            cache_indices: None,
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
                        tile_indices: NodeLayerIndices::HeightNormal(child.0, child.0),
                        cache_indices: None,
                    };

                    nodes.push(child_node);
                    nodes[parent].children[i] = Some(child);
                    pending.push_back(child);
                }
            }
        }

        Self {
            nodes,
            heights: TileCache::new(1024),
            normals: TileCache::new(512),
            splats: TileCache::new(96),
        }
    }

    fn update_priorities(&mut self, camera: Point3<f32>) {
        for node in self.nodes.iter_mut() {
            node.priority = Priority(
                node.bounds.square_distance(camera).max(1.0) /
                    (node.min_distance * node.min_distance),
            );
        }
        self.heights.update_priorities(&mut self.nodes);
        self.normals.update_priorities(&mut self.nodes);
        self.splats.update_priorities(&mut self.nodes);
    }

    pub fn update(&mut self, camera: Point3<f32>) {
        self.update_priorities(camera);

        let mut pending = VecDeque::new();
        pending.push_back(NodeId::root());
        while let Some(id) = pending.pop_front() {
            if self.nodes[id].priority < Priority::cutoff() {
                continue;
            }

            if self.nodes[id].cache_indices.is_none() {
                let elem = (self.nodes[id].priority, id);
                match self.nodes[id].tile_indices {
                    NodeLayerIndices::Height(..) => {
                        self.heights.add_missing(elem);
                    }
                    NodeLayerIndices::HeightNormal(..) => {
                        self.heights.add_missing(elem);
                        self.normals.add_missing(elem);
                    }
                    NodeLayerIndices::HeightNormalSplat(..) => {
                        self.heights.add_missing(elem);
                        self.normals.add_missing(elem);
                        self.splats.add_missing(elem);
                    }
                }
            }

            for i in 0..4 {
                if let Some(child) = self.nodes[id].children[i] {
                    pending.push_back(child);
                }
            }
        }

        self.heights.load_missing(&mut self.nodes);
        self.normals.load_missing(&mut self.nodes);
        self.splats.load_missing(&mut self.nodes);
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }
}
