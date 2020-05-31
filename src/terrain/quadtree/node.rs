use crate::generate::EARTH_CIRCUMFERENCE;
use crate::terrain::tile_cache::Priority;
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
    fn new(level: u8, face: u8, x: u32, y: u32) -> Self {
        debug_assert!(face < 6);
        debug_assert!(level < 26);
        debug_assert!(x <= 0x3ffffff);
        debug_assert!(y <= 0x3ffffff);
        Self((level as u64) << 56 | (face as u64) << 53 | (y as u64) << 26 | (x as u64))
    }
    fn roots() -> [Self; 6] {
        [
            Self::new(0, 0, 0, 0),
            Self::new(0, 1, 0, 0),
            Self::new(0, 2, 0, 0),
            Self::new(0, 3, 0, 0),
            Self::new(0, 4, 0, 0),
            Self::new(0, 5, 0, 0),
        ]
    }

    pub fn x(&self) -> u32 {
        self.0 as u32 & 0x3ffffff
    }
    pub fn y(&self) -> u32 {
        (self.0 >> 26) as u32 & 0x3ffffff
    }
    pub fn level(&self) -> u8 {
        (self.0 >> 56) as u8
    }
    pub fn face(&self) -> u8 {
        (self.0 >> 53) as u8 & 0x7
    }

    pub fn aprox_side_length(&self) -> f32 {
        ROOT_SIDE_LENGTH / (1u32 << self.level()) as f32
    }

    /// Minimum distance from the center of this node on the face of a cube with coordinates from
    /// [-1, 1].
    pub fn min_distance(&self) -> f64 {
        4.0 / (1u32 << self.level()) as f64
    }

    fn center_cspace(&self) -> Vector3<f64> {
        self.cell_position_cspace(0, 0, 0, 1)
    }

    fn fspace_to_cspace(&self, x: f64, y: f64) -> Vector3<f64> {
        match self.face() {
            0 => Vector3::new(1.0, x, -y),
            1 => Vector3::new(-1.0, -x, -y),
            2 => Vector3::new(x, 1.0, y),
            3 => Vector3::new(-x, -1.0, y),
            4 => Vector3::new(x, -y, 1.0),
            5 => Vector3::new(-x, -y, -1.0),
            _ => unreachable!(),
        }
    }

    /// Interpolate position on this node assuming a grid with given `resolution` and surrounded by
    /// `skirt` cells outside the borders on each edge (but counted in resolution). Assumes [grid
    /// registration](https://www.ngdc.noaa.gov/mgg/global/gridregistration.html). Used for
    /// elevation data.
    ///
    ///       |       |
    ///     +---+---+---+
    ///   --|   |   |   |--
    ///     +---+---+---+
    ///     |   |   |   |
    ///     +---+---+---+
    ///   --|   |   |   |--
    ///     +---+---+---+
    ///       |       |
    ///
    pub fn grid_position_cspace(&self, x: i32, y: i32, skirt: u16, resolution: u16) -> Vector3<f64> {
        let fx = (x - skirt as i32) as f64 / (resolution - 1 - 2 * skirt) as f64;
        let fy = (y - skirt as i32) as f64 / (resolution - 1 - 2 * skirt) as f64;
        let scale = 2.0 / (1u32 << self.level()) as f64;

        let fx = (self.x() as f64 + fx) * scale - 1.0;
        let fy = (self.y() as f64 + fy) * scale - 1.0;
        self.fspace_to_cspace(fx, fy)
    }

    /// Same as `position_cspace_corners` but uses "cell registration". Used for textures/normalmaps.
    ///
    ///     |       |
    ///   --+---+---+--
    ///     |   |   |
    ///     +---+---+
    ///     |   |   |
    ///   --+---+---+--
    ///     |       |
    ///
    pub fn cell_position_cspace(&self, x: i32, y: i32, skirt: u16, resolution: u16) -> Vector3<f64> {
        let fx = ((x - skirt as i32) as f64 + 0.5) / (resolution - 2 * skirt) as f64;
        let fy = ((y - skirt as i32) as f64 + 0.5) / (resolution - 2 * skirt) as f64;
        let scale = 2.0 / (1u32 << self.level()) as f64;

        let fx = (self.x() as f64 + fx) * scale - 1.0;
        let fy = (self.y() as f64 + fy) * scale - 1.0;
        self.fspace_to_cspace(fx, fy)
    }

    /// How much this node is needed for the current frame. Nodes with priority less than 1.0 will
    /// not be rendered (they are too detailed).
    pub fn priority(&self, camera_cspace: Point3<f64>) -> Priority {
        let min_distance = self.min_distance();
        let center = self.center_cspace();
        let r = 1.0 / (1u64 << (self.level())) as f64;
        let dx = ((center.x - r) - camera_cspace.x).max(0.0).max(camera_cspace.x - (center.x + r));
        let dy = ((center.y - r) - camera_cspace.y).max(0.0).max(camera_cspace.y - (center.y + r));
        let dz = ((center.z - r) - camera_cspace.z).max(0.0).max(camera_cspace.z - (center.z + r));
        let distance = dx * dx + dy * dy + dz * dz;

        Priority::from_f32(((min_distance * min_distance) / distance.max(1e-8)) as f32)
    }

    pub fn parent(&self) -> Option<(VNode, u8)> {
        if self.level() == 0 {
            return None;
        }
        let child_index = ((self.x() % 2) + (self.y() % 2) * 2) as u8;
        Some((VNode::new(self.level() - 1, self.face(), self.x() / 2, self.y() / 2), child_index))
    }

    pub fn children(&self) -> [VNode; 4] {
        assert!(self.level() < 31);
        [
            VNode::new(self.level() + 1, self.face(), self.x() * 2, self.y() * 2),
            VNode::new(self.level() + 1, self.face(), self.x() * 2 + 1, self.y() * 2),
            VNode::new(self.level() + 1, self.face(), self.x() * 2, self.y() * 2 + 1),
            VNode::new(self.level() + 1, self.face(), self.x() * 2 + 1, self.y() * 2 + 1),
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
            node = VNode::new(node.level() - 1, node.face(), node.x() / 2, node.y() / 2);
        }
        Some((node, generations, offset))
    }

    pub fn breadth_first<Visit>(mut visit: Visit)
    where
        Visit: FnMut(VNode) -> bool,
    {
        let mut pending = VecDeque::new();
        for &n in &Self::roots() {
            if visit(n) {
                pending.push_back(n);
            }
        }
        while let Some(node) = pending.pop_front() {
            for &child in node.children().iter() {
                if visit(child) {
                    pending.push_back(child);
                }
            }
        }
    }

    pub fn make_nodes(max_level: u8) -> Vec<VNode> {
        let mut nodes = Self::roots().to_vec();
        let mut pending: VecDeque<VNode> = nodes.iter().cloned().collect();

        while let Some(parent) = pending.pop_front() {
            if parent.level() >= max_level {
                continue;
            }

            for &child in parent.children().iter() {
                // let distance = (child.center_cspace() - Vector3::new(0.0,1.0,0.0)).magnitude() as f32;
                // let min_distance = 1.95 / (1 << child.level()) as f32;

                if child.level() <= 3 {
                    nodes.push(child);
                    pending.push_back(child);
                }
            }
        }

        nodes
    }
}
