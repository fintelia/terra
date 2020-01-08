use cgmath::*;
use serde::{Deserialize, Serialize};

use std::collections::VecDeque;

use crate::terrain::quadtree::NodeId;
use crate::terrain::tile_cache::Priority;

use crate::utils::math::BoundingBox;

const ROOT_SIDE_LENGTH: f32 = 4194304.0;

lazy_static! {
    pub static ref OFFSETS: [Vector2<i32>; 4] =
        [Vector2::new(0, 0), Vector2::new(1, 0), Vector2::new(0, 1), Vector2::new(1, 1),];
    pub static ref CENTER_OFFSETS: [Vector2<i32>; 4] =
        [Vector2::new(-1, -1), Vector2::new(1, -1), Vector2::new(-1, 1), Vector2::new(1, 1),];
}

#[derive(Serialize, Deserialize)]
pub(crate) struct Node {
    pub level: u8,
    /// Tuple of this node's parent id, and the index of this node in its parents child list.
    pub parent: Option<(NodeId, u8)>,
    pub children: [Option<NodeId>; 4],

    pub bounds: BoundingBox,
}
impl Node {
    pub fn side_length(&self) -> f32 {
        ROOT_SIDE_LENGTH / (1u32 << self.level) as f32
    }
    pub fn min_distance(&self) -> f32 {
        self.side_length() * 1.95
    }

    /// How much this node is needed for the current frame. Nodes with priority less than 1.0 will
    /// not be rendered (they are too detailed).
    pub fn priority(&self, camera: Point3<f32>) -> Priority {
        let min_distance = self.min_distance();
        Priority::from_f32(
            (min_distance * min_distance) / self.bounds.square_distance_xz(camera).max(0.001),
        )
    }

    pub fn make_nodes(playable_radius: f32, max_level: u8) -> Vec<Node> {
        let node = Node {
            level: 0,
            parent: None,
            children: [None; 4],
            bounds: BoundingBox {
                min: Point3::new(-ROOT_SIDE_LENGTH * 0.5, 0.0, -ROOT_SIDE_LENGTH * 0.5),
                max: Point3::new(ROOT_SIDE_LENGTH * 0.5, 0.0, ROOT_SIDE_LENGTH * 0.5),
            },
        };

        let mut nodes = vec![node];
        let mut pending = VecDeque::new();
        pending.push_back(NodeId::root());

        while let Some(parent) = pending.pop_front() {
            if nodes[parent].level >= max_level {
                continue;
            }

            let min_distance = nodes[parent].min_distance() * 0.5;
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

            for i in 0..4 {
                if bounds[i].distance(Point3::origin()) < playable_radius + min_distance {
                    let child = NodeId::new(nodes.len() as u32);
                    let child_node = Node {
                        level: nodes[parent].level + 1,
                        parent: Some((parent, i as u8)),
                        children: [None; 4],
                        bounds: bounds[i],
                    };

                    nodes.push(child_node);
                    nodes[parent].children[i] = Some(child);
                    pending.push_back(child);
                }
            }
        }

        nodes
    }

    pub fn find_ancestor<Visit>(
        nodes: &Vec<Node>,
        mut node: NodeId,
        mut visit: Visit,
    ) -> Option<(NodeId, usize, Vector2<i32>)>
    where
        Visit: FnMut(NodeId) -> bool,
    {
        let mut generations = 0;
        let mut offset = Vector2::new(0, 0);
        while !visit(node) {
            match nodes[node].parent {
                None => return None,
                Some((parent, child_index)) => {
                    node = parent;
                    offset += OFFSETS[child_index as usize] * (1 << generations);
                    generations += 1;
                }
            }
        }
        Some((node, generations, offset))
    }
}
