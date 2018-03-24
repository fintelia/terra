#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use std::f64::consts::PI;
use cgmath::{ElementWise, InnerSpace, Vector2, Vector3};

use sky::lut::{LookupTable, LookupTableDefinition};

// Simulation is done at λ = (680, 550, 440) nm = (red, green, blue).
// See https://hal.inria.fr/inria-00288758/document

const Rg: f64 = 6371000.0;
const Rt: f64 = 6471000.0;

mod rayleigh {
    use super::*;

    // For rayleigh scattering there is no absorbsion so βe = βs.
    pub const βe: Vector3<f64> = Vector3 {
        x: 5.8e-6,
        y: 13.5e-6,
        z: 33.1e-6,
    };
    pub const βs: Vector3<f64> = βe;
    pub const H: f64 = 8000.0;

    pub fn P(µ: f64) -> f64 {
        3.0 / (16.0 * PI) * (1.0 + µ * µ)
    }
}

mod mie {
    use super::*;

    pub const βs: f64 = 2.0e-5;
    pub const βe: f64 = βs / 0.9;
    pub const H: f64 = 1200.0;

    #[allow(unused)]
    pub const g: f64 = 0.76;

    pub fn P(µ: f64) -> f64 {
        3.0 / (8.0 * PI) * ((1.0 - g * g) * (1.0 + µ * µ))
            / ((2.0 + g * g) * f64::powf(1.0 + g * g - 2.0 * g * µ, 1.5))
    }
}

fn integral<F: Fn(Vector2<f64>) -> Vector3<f64>>(
    r: f64,
    theta: f64,
    steps: u32,
    check_planet_surface: bool,
    f: F,
) -> Vector3<f64> {
    let b = 2.0 * r * f64::cos(theta);
    let c_atmosphere = r * r - Rt * Rt;
    let c_ground = r * r - Rg * Rg;
    let length = if check_planet_surface && b < 0.0 && b * b - 4.0 * c_ground >= 0.0 {
        (-b - f64::sqrt(b * b - 4.0 * c_ground)) / 2.0
    } else {
        (-b + f64::sqrt(b * b - 4.0 * c_atmosphere)) / 2.0
    };
    let step_length = length / f64::from(steps);

    let x = Vector2::new(0.0, r);
    let v = Vector2::new(f64::sin(theta), f64::cos(theta)) * step_length;

    let mut sum = Vector3::new(0.0, 0.0, 0.0);
    for i in 0..steps {
        let y = x + v * (f64::from(i) + 0.5);
        sum += f(y) * step_length;
    }

    sum
}

pub(super) struct TransmittanceTable {
    pub steps: u32,
}
impl LookupTableDefinition for TransmittanceTable {
    fn filename(&self) -> String {
        format!(
            "sky/transmittance.{}x{}.{:02}.bin",
            self.size()[0],
            self.size()[1],
            self.steps
        )
    }
    fn size(&self) -> [u16; 4] {
        [64, 256, 1, 1]
    }
    fn compute(&self, [x, y, _, _]: [u16; 4]) -> [f32; 4] {
        let xx = f64::from(x) / f64::from(self.size()[0] - 1);
        let yy = f64::from(y) / f64::from(self.size()[1] - 1);

        let r = Rg + (Rt - Rg) * xx;
        let v = 2.0 * yy - 1.0;

        let t = integral(r, f64::acos(v), self.steps, false, |y| {
            let height = y.magnitude() - Rg;
            let βe_R = rayleigh::βe * f64::exp(-height / rayleigh::H);
            let βe_M = mie::βe * f64::exp(-height / mie::H);
            βe_R + Vector3::new(βe_M, βe_M, βe_M)
        });

        [
            f64::exp(-t.x) as f32,
            f64::exp(-t.y) as f32,
            f64::exp(-t.z) as f32,
            0.0,
        ]
    }
}

pub(super) struct InscatteringTable {
    pub steps: u32,
    pub transmittance: LookupTable,
}
impl InscatteringTable {
    fn compute_parameters(u_r: f64, u_µ: f64, u_µ_s: f64, u_v: f64) -> (f64, f64, f64, f64) {
        // See: https://github.com/ebruneton/precomputed_atmospheric_scattering/blob/fb903786e59d3cf21ebc107edee5768366bdad5e/atmosphere/functions.glsl#L832
        assert!(u_r >= 0.0 && u_r <= 1.0);
        assert!(u_µ >= 0.0 && u_µ <= 1.0);
        assert!(u_µ_s >= 0.0 && u_µ_s <= 1.0);
        assert!(u_v >= 0.0 && u_v <= 1.0);

        let H = f64::sqrt(Rt * Rt - Rg * Rg);
        let ρ = u_r * H;

        let r = f64::sqrt(ρ * ρ + Rg * Rg);

        let µ = if u_µ < 0.5 {
            // Ray intersects ground
            let d_min = r - Rg;
            let d_max = ρ;
            let d = d_min + (d_max - d_min) * (1.0 - 2.0 * u_µ);
            if d == 0.0 {
                -1.0
            } else {
                let µ = -(ρ * ρ + d * d) / (2.0 * r * d);
                µ.max(-1.0).min(1.0)
            }
        } else {
            // Ray does not intersect ground
            let d_min = Rt - r;
            let d_max = ρ + H;
            let d = d_min + (d_max - d_min) * (2.0 * u_µ - 1.0);
            if d == 0.0 {
                1.0
            } else {
                let µ = (H * H - ρ * ρ - d * d) / (2.0 * r * d);
                µ.max(-1.0).min(1.0)
            }
        };

        let µ_s = (-(f64::ln(1.0 - u_µ_s * (1.0 - f64::exp(-3.6))) + 0.6) / 3.0)
            .max(-1.0)
            .min(1.0);
        let v = 2.0 * u_v - 1.0;

        (r, µ, µ_s, v)
    }
    #[cfg(test)]
    fn reverse_parameters(r: f64, µ: f64, µ_s: f64, v: f64) -> (f64, f64, f64, f64) {
        assert!(r >= Rg && r <= Rt);
        assert!(µ >= -1.0 && µ <= 1.0);
        assert!(µ_s >= -1.0 && µ_s <= 1.0);
        assert!(v >= -1.0 && v <= 1.0);

        let H = f64::sqrt(Rt * Rt - Rg * Rg);
        let ρ = f64::sqrt(r * r - Rg * Rg);
        let Δ = r * r * µ * µ - ρ * ρ;

        let u_r = ρ / H;

        let u_µ = if µ < 0.0 && Δ >= 0.0 {
            // Ray intersects planet surface.
            let d_min = r - Rg;
            let d_max = ρ;
            let d = -r * µ - f64::sqrt(Δ);
            if d_min == d_max {
                0.0
            } else {
                0.5 - 0.5 * (d - d_min) / (d_max - d_min)
            }
        } else {
            let d_min = Rt - r;
            let d_max = ρ + H;
            let d = -r * µ + f64::sqrt(Δ + H * H);

            0.5 + 0.5 * (d - d_min) / (d_max - d_min)
        };
        let u_µ_s = (1.0 - f64::exp(-3.0 * µ_s - 0.6)) / (1.0 - f64::exp(-3.6));
        let u_v = (1.0 + v) / 2.0;

        (u_r, u_µ, u_µ_s, u_v)
    }
}
impl LookupTableDefinition for InscatteringTable {
    fn filename(&self) -> String {
        format!(
            "sky/inscattering.{}x{}x{}x{}.{:02}.bin",
            self.size()[0],
            self.size()[1],
            self.size()[2],
            self.size()[3],
            self.steps
        )
    }
    fn size(&self) -> [u16; 4] {
        [32, 128, 32, 8]
    }
    fn compute(&self, [x, y, z, w]: [u16; 4]) -> [f32; 4] {
        let (r, µ, µ_s, v) = Self::compute_parameters(
            f64::from(x) / f64::from(self.size()[0] - 1),
            f64::from(y) / f64::from(self.size()[1] - 1),
            f64::from(z) / f64::from(self.size()[2] - 1),
            f64::from(w) / f64::from(self.size()[3] - 1),
        );

        let L_sun = 1.0;
        let s = integral(r, f64::acos(µ), self.steps, true, |y| {
            let h = y.magnitude() - Rg;

            let xx = (h / (Rt - Rg)) as f32;
            let yy = (µ_s * 0.5 + 0.5) as f32;
            let [Tr, Tg, Tb, _] = self.transmittance.get2(xx, yy);
            let T = Vector3::new(Tr as f64, Tg as f64, Tb as f64);

            let R = T.mul_element_wise(rayleigh::βs) * f64::exp(-h / rayleigh::H) * rayleigh::P(v)
                * L_sun;
            let M = T * mie::βs * f64::exp(-h / mie::H) * mie::P(v) * L_sun;
            R + M
        });
        [s.x as f32, s.y as f32, s.z as f32, 0.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{self, Rng};
    use sky::lut::LookupTableDefinition;
    use cache::{AssetLoadContext, GeneratedAsset};

    #[test]
    fn invert_inscatter_parameters() {
        let mut rng = rand::thread_rng();
        for i in 0..100 {
            let (x, y, z, w) = (
                rng.gen_range(0.0, 1.0),
                rng.gen_range(0.0, 1.0),
                rng.gen_range(0.0, 1.0),
                rng.gen_range(0.0, 1.0),
            );

            let (r, µ, µ_s, v) = InscatteringTable::compute_parameters(x, y, z, w);
            let (x2, y2, z2, w2) = InscatteringTable::reverse_parameters(r, µ, µ_s, v);

            assert_relative_eq!(x, x2, max_relative = 0.0001);
            assert_relative_eq!(w, w2, max_relative = 0.0001);
            assert_relative_eq!(z, z2, max_relative = 0.0001);
            assert_relative_eq!(y, y2, max_relative = 0.0001);
        }
    }

    #[test]
    fn transmittance_enough_steps() {
        let t1 = TransmittanceTable { steps: 10000 };
        let t2 = TransmittanceTable { steps: 20000 };

        let t1 = t1.load(&mut AssetLoadContext::new()).unwrap();
        let t2 = t2.load(&mut AssetLoadContext::new()).unwrap();
        assert_eq!(t1.size, t2.size);

        for x in 0..t1.size[0] as usize {
            for y in 0..t1.size[1] as usize {
                let v1 = t1.data[x + y * t1.size[0] as usize];
                let v2 = t2.data[x + y * t2.size[0] as usize];
                for i in 0..4 {
                    assert_relative_eq!(v1[i], v2[i], max_relative = 0.05);
                }
            }
        }
    }
}
