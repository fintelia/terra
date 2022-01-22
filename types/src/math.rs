use cgmath::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct BoundingBox {
    pub min: Point3<f32>,
    pub max: Point3<f32>,
}

impl BoundingBox {
    #[allow(unused)]
    pub fn new(min: Point3<f32>, max: Point3<f32>) -> Self {
        Self { min, max }
    }
    #[allow(unused)]
    pub fn distance(&self, p: Point3<f32>) -> f32 {
        self.square_distance(p).sqrt()
    }
    #[allow(unused)]
    pub fn square_distance(&self, p: Point3<f32>) -> f32 {
        let dx = (self.min.x - p.x).max(0.0).max(p.x - self.max.x);
        let dy = (self.min.y - p.y).max(0.0).max(p.y - self.max.y);
        let dz = (self.min.z - p.z).max(0.0).max(p.z - self.max.z);
        dx * dx + dy * dy + dz * dz
    }

    #[allow(unused)]
    pub fn square_distance_xz(&self, p: Point3<f32>) -> f32 {
        let dx = (self.min.x - p.x).max(0.0).max(p.x - self.max.x);
        let dz = (self.min.z - p.z).max(0.0).max(p.z - self.max.z);
        dx * dx + dz * dz
    }
}

#[derive(Clone, Debug)]
pub struct InfiniteFrustum {
    pub planes: [Vector4<f64>; 5],
}
impl InfiniteFrustum {
    fn normalize_plane(plane: Vector4<f64>) -> Vector4<f64> {
        let magnitude = (plane.x * plane.x + plane.y * plane.y + plane.z * plane.z).sqrt();
        plane / magnitude
    }

    pub fn from_matrix(m: Matrix4<f64>) -> Self {
        let m = m.transpose();
        Self {
            planes: [
                Self::normalize_plane(m.w + m.x),
                Self::normalize_plane(m.w - m.x),
                Self::normalize_plane(m.w + m.y),
                Self::normalize_plane(m.w - m.y),
                Self::normalize_plane(m.w + m.z),
            ],
        }
    }

    pub fn intersects_sphere(&self, center: Vector3<f64>, radius_squared: f64) -> bool {
        for p in &self.planes[0..5] {
            let distance = p.x * center.x + p.y * center.y + p.z * center.z + p.w;
            if distance < 0.0 && distance * distance > radius_squared {
                return false;
            }
        }
        true
    }
}
