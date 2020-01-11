use cgmath::*;
use serde::{Deserialize, Serialize};

use std::collections::VecDeque;

use crate::terrain::tile_cache::Priority;

use crate::utils::math::BoundingBox;

const ROOT_SIDE_LENGTH: f32 = 4194304.0;

lazy_static! {
    pub static ref OFFSETS: [Vector2<i32>; 4] =
        [Vector2::new(0, 0), Vector2::new(1, 0), Vector2::new(0, 1), Vector2::new(1, 1),];
    pub static ref CENTER_OFFSETS: [Vector2<i32>; 4] =
        [Vector2::new(-1, -1), Vector2::new(1, -1), Vector2::new(-1, 1), Vector2::new(1, 1),];
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash, Serialize, Deserialize)]
pub(crate) struct VNode {
    level: u8,
    x: u32,
    y: u32,
}
impl VNode {
    pub fn root() -> Self {
        Self { level: 0, x: 0, y: 0 }
    }

    pub fn level(&self) -> u8 {
        self.level
    }

    pub fn side_length(&self) -> f32 {
        ROOT_SIDE_LENGTH / (1u32 << self.level) as f32
    }
    pub fn min_distance(&self) -> f32 {
        self.side_length() * 1.95
    }

    pub fn bounds(&self) -> BoundingBox {
        let side_length = self.side_length();
        let min = Point3::new(
            -ROOT_SIDE_LENGTH * 0.5 + side_length * self.x as f32,
            0.0,
            -ROOT_SIDE_LENGTH * 0.5 + side_length * self.y as f32,
        );
        let max = Point3::new(min.x + side_length, 8000.0, min.z + side_length);
        BoundingBox { min, max }
    }

    /// How much this node is needed for the current frame. Nodes with priority less than 1.0 will
    /// not be rendered (they are too detailed).
    pub fn priority(&self, camera: Point3<f32>) -> Priority {
        let min_distance = self.min_distance();
        Priority::from_f32(
            (min_distance * min_distance) / self.bounds().square_distance_xz(camera).max(0.001),
        )
    }

    pub fn parent(&self) -> Option<(VNode, u8)> {
        if self.level == 0 {
            return None;
        }
        let child_index = ((self.x % 2) + (self.y % 2) * 2) as u8;
        Some((VNode { level: self.level - 1, x: self.x / 2, y: self.y / 2 }, child_index))
    }

    pub fn children(&self) -> [VNode; 4] {
        assert!(self.level < 31);
        [
            VNode { level: self.level + 1, x: self.x * 2, y: self.y * 2 },
            VNode { level: self.level + 1, x: self.x * 2 + 1, y: self.y * 2 },
            VNode { level: self.level + 1, x: self.x * 2, y: self.y * 2 + 1 },
            VNode { level: self.level + 1, x: self.x * 2 + 1, y: self.y * 2 + 1 },
        ]
    }

    pub fn find_ancestor<Visit>(&self, mut visit: Visit) -> Option<(VNode, usize, Vector2<u32>)>
    where
        Visit: FnMut(VNode) -> bool,
    {
        let mut node = *self;
        let mut generations = 0;
        let mut offset = Vector2::new(0, 0);
        while !visit(node) {
            if node.level == 0 {
                return None;
            }
            offset += Vector2::new(node.x & 1, node.y & 1) * (1 << generations);
            generations += 1;
            node = VNode { level: node.level - 1, x: node.x / 2, y: node.y / 2 };
        }
        Some((node, generations, offset))
    }


    pub fn breadth_first<Visit>(mut visit: Visit)
    where
        Visit: FnMut(VNode) -> bool,
    {
        let mut pending = VecDeque::new();
        if visit(VNode::root()) {
            pending.push_back(VNode::root());
        }
        while let Some(node) = pending.pop_front() {
            for &child in node.children().iter() {
                if visit(child) {
                    pending.push_back(child);
                }
            }
        }
    }

    pub fn make_nodes(playable_radius: f32, max_level: u8) -> Vec<VNode> {
        let mut nodes = vec![Self::root()];
        let mut pending = VecDeque::new();
        pending.push_back(Self::root());

        while let Some(parent) = pending.pop_front() {
            if parent.level >= max_level {
                continue;
            }

            for &child in parent.children().iter() {
                if child.bounds().distance(Point3::origin())
                    < playable_radius + child.min_distance()
                {
                    nodes.push(child);
                    pending.push_back(child);
                }
            }
        }

        nodes
    }
}
