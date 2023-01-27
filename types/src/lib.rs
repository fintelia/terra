#[macro_use]
extern crate lazy_static;

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

mod math;
mod node;

pub use math::{BoundingBox, InfiniteFrustum};
pub use node::{VNode, NODE_OFFSETS};

pub const EARTH_RADIUS: f64 = 6371000.0;
pub const EARTH_CIRCUMFERENCE: f64 = 2.0 * PI * EARTH_RADIUS;
pub const EARTH_SEMIMAJOR_AXIS: f64 = 6378137.0;
pub const EARTH_SEMIMINOR_AXIS: f64 = 6356752.314245;
pub const ROOT_SIDE_LENGTH: f32 = (EARTH_CIRCUMFERENCE * 0.25) as f32;
pub const MAX_QUADTREE_LEVEL: u8 = VNode::LEVEL_CELL_5MM;

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Priority(f32);
impl Priority {
    pub fn cutoff() -> Self {
        Priority(1.0)
    }
    pub fn none() -> Self {
        Priority(-1.0)
    }
    pub fn from_f32(value: f32) -> Self {
        assert!(value.is_finite());
        Priority(value)
    }
}
impl Eq for Priority {}
impl Ord for Priority {
    fn cmp(&self, other: &Self) -> ::std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

pub struct VFace(pub u8);
impl std::fmt::Display for VFace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(
            f,
            "{}",
            match self.0 {
                0 => "0E",
                1 => "180E",
                2 => "90E",
                3 => "90W",
                4 => "N",
                5 => "S",
                _ => unreachable!(),
            }
        )
    }
}

pub struct VSector(pub VFace, pub u8, pub u8);
impl std::fmt::Display for VSector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "S-{}-x{:03}-y{:03}", self.0, self.1, self.2)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
