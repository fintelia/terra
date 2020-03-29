use crate::generate::EARTH_CIRCUMFERENCE;
use crate::terrain::tile_cache::Priority;
use crate::utils::math::BoundingBox;
use cgmath::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

const ROOT_SIDE_LENGTH: f32 = (EARTH_CIRCUMFERENCE * 0.25) as f32;

lazy_static! {
    pub static ref OFFSETS: [Vector2<i32>; 4] =
        [Vector2::new(0, 0), Vector2::new(1, 0), Vector2::new(0, 1), Vector2::new(1, 1),];
    pub static ref CENTER_OFFSETS: [Vector2<i32>; 4] =
        [Vector2::new(-1, -1), Vector2::new(1, -1), Vector2::new(-1, 1), Vector2::new(1, 1),];
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash, Serialize, Deserialize)]
pub(crate) struct VNode(u64);

impl VNode {
    fn new(level: u8, x: u32, y: u32) -> Self {
        Self((level as u64) << 56 | (y as u64) << 24 | (x as u64))
    }
    pub fn x(&self) -> u32 {
        self.0 as u32 & 0xffffff
    }
    pub fn y(&self) -> u32 {
        (self.0 >> 24) as u32 & 0xffffff
    }
    pub fn level(&self) -> u8 {
        (self.0 >> 56) as u8
    }

    pub fn root() -> Self {
        Self::new(0, 0, 0)
    }

    pub fn aprox_side_length(&self) -> f32 {
        ROOT_SIDE_LENGTH / (1u32 << self.level()) as f32
    }

    // Minimum distance from the center of this node on the face of a cube with coordinates from [-1,
    // 1].
    pub fn min_distance(&self) -> f64 {
        1.95 / (1u32 << self.level()) as f64
    }

    pub fn bounds(&self) -> BoundingBox {
        let side_length = self.aprox_side_length();
        let min = Point3::new(
            -ROOT_SIDE_LENGTH * 0.5 + side_length * self.x() as f32,
            0.0,
            -ROOT_SIDE_LENGTH * 0.5 + side_length * self.y() as f32,
        );
        let max = Point3::new(min.x + side_length, 8000.0, min.z + side_length);
        BoundingBox { min, max }
    }

    fn center_cspace(&self) -> Point3<f64> {
        // TODO: adjust for which face this is
        Point3::new(
            (self.x() * 2 + 1) as f64 / (1u32 << self.level()) as f64 - 1.0,
            1.0,
            (self.y() * 2 + 1) as f64 / (1u32 << self.level()) as f64 - 1.0,
        )
    }

    /// How much this node is needed for the current frame. Nodes with priority less than 1.0 will
    /// not be rendered (they are too detailed).
    pub fn priority(&self, camera_cspace: Point3<f64>) -> Priority {
        let min_distance = self.min_distance();
        Priority::from_f32(
            ((min_distance * min_distance) / (self.center_cspace().distance2(camera_cspace) -
                                              1.41 / (1u64 << (self.level()*2)) as f64).max(1e-9)) as f32,
        )
    }

    pub fn parent(&self) -> Option<(VNode, u8)> {
        if self.level() == 0 {
            return None;
        }
        let child_index = ((self.x() % 2) + (self.y() % 2) * 2) as u8;
        Some((VNode::new(self.level() - 1, self.x() / 2, self.y() / 2), child_index))
    }

    pub fn children(&self) -> [VNode; 4] {
        assert!(self.level() < 31);
        [
            VNode::new(self.level() + 1, self.x() * 2, self.y() * 2),
            VNode::new(self.level() + 1, self.x() * 2 + 1, self.y() * 2),
            VNode::new(self.level() + 1, self.x() * 2, self.y() * 2 + 1),
            VNode::new(self.level() + 1, self.x() * 2 + 1, self.y() * 2 + 1),
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
            if node.level() == 0 {
                return None;
            }
            offset += Vector2::new(node.x() & 1, node.y() & 1) * (1 << generations);
            generations += 1;
            node = VNode::new(node.level() - 1, node.x() / 2, node.y() / 2);
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
            if parent.level() >= max_level {
                continue;
            }

            for &child in parent.children().iter() {
                if child.bounds().distance(Point3::origin())
                    < playable_radius + child.aprox_side_length() * 1.95
                {
                    nodes.push(child);
                    pending.push_back(child);
                }
            }
        }

        nodes
    }
}
