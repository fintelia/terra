use cgmath::*;

use std::collections::VecDeque;

use super::{NodeId, NUM_LAYERS};
use terrain::tile_cache::Priority;

use utils::math::BoundingBox;

pub(crate) struct Node {
    pub level: u8,
    #[allow(unused)]
    pub parent: Option<NodeId>,
    pub children: [Option<NodeId>; 4],

    pub bounds: BoundingBox,
    pub side_length: f32,
    pub min_distance: f32,

    pub size: i32,
    pub center: Point2<i32>,

    /// Index of this node in the tile list.
    pub tile_indices: [Option<u32>; NUM_LAYERS],

    /// How much this node is needed for the current frame. Nodes with priority less than 1.0 will
    /// not be rendered (they are too detailed).
    pub priority: Priority,

    pub visible: bool,
}
impl Node {
    pub fn priority(&self) -> Priority {
        self.priority
    }

    pub fn make_nodes(side_length: f32, playable_radius: f32, max_level: u8) -> Vec<Node> {
        let node = Node {
            level: 0,
            parent: None,
            children: [None; 4],
            bounds: BoundingBox {
                min: Point3::new(-side_length * 0.5, 0.0, -side_length * 0.5),
                max: Point3::new(side_length * 0.5, 0.0, side_length * 0.5),
            },
            side_length,
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
                    let child = NodeId::new(nodes.len() as u32);
                    let child_node = Node {
                        level: nodes[parent].level + 1,
                        parent: Some(parent),
                        children: [None; 4],
                        bounds: bounds[i],
                        side_length: nodes[parent].side_length * 0.5,
                        min_distance: nodes[parent].min_distance * 0.5,
                        size: nodes[parent].size / 2,
                        center: Point2::from_vec(
                            nodes[parent].center.to_vec() -
                                offsets[i] * (nodes[parent].size / 4),
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

        nodes
    }
}
