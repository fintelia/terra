#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use cgmath::{ElementWise, InnerSpace, Vector2, Vector3, Vector4, VectorSpace, Zero};
use std::f64::consts::PI;

use sky::lut::{LookupTable, LookupTableDefinition};

// Simulation is done at λ = (680, 550, 440) nm = (red, green, blue).
// See https://hal.inria.fr/inria-00288758/document
// https://media.contentapi.ea.com/content/dam/eacom/frostbite/files/s2016-pbs-frostbite-sky-clouds-new.pdf

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

    #[allow(unused)]
    pub fn P(µ: f64) -> f64 {
        3.0 / (16.0 * PI) * (1.0 + µ * µ)
    }
}

mod mie {
    use super::*;

    pub const βs: f64 = 2.0e-6;
    pub const βe: f64 = βs / 0.9;
    pub const H: f64 = 1200.0;
    pub const g: f64 = 0.76;

    #[allow(unused)]
    pub fn P(µ: f64) -> f64 {
        3.0 / (8.0 * PI) * ((1.0 - g * g) * (1.0 + µ * µ))
            / ((2.0 + g * g) * f64::powf(1.0 + g * g - 2.0 * g * µ, 1.5))
    }
}

fn integral<V, F>(r: f64, theta: f64, steps: u32, force_hit_planet_surface: bool, f: F) -> V
where
    V: VectorSpace<Scalar = f64>,
    F: Fn(Vector2<f64>) -> V,
{
    let b = 2.0 * r * f64::cos(theta);
    let c_atmosphere = r * r - Rt * Rt;
    let c_ground = r * r - Rg * Rg;
    let length = if force_hit_planet_surface {
        if b * b - 4.0 * c_ground >= 0.0 {
            (-b - f64::sqrt(b * b - 4.0 * c_ground)) / 2.0
        } else {
            // Doesn't actually hit planet surface. Fake it by taking closest point.
            -b / 2.0
        }
    } else {
        (-b + f64::sqrt(b * b - 4.0 * c_atmosphere)) / 2.0
    };

    assert!(!r.is_nan());
    assert!(!theta.is_nan());
    assert!(!length.is_nan());

    if length <= 0.0 {
        return Zero::zero();
    }

    let step_length = length / f64::from(steps);

    let x = Vector2::new(0.0, r);
    let v = Vector2::new(f64::sin(theta), f64::cos(theta)) * step_length;

    let mut sum = V::zero();
    for i in 0..steps {
        let y = x + v * (f64::from(i) + 0.5);
        sum = sum + f(y) * step_length;
    }

    sum
}

pub(super) struct TransmittanceTable {
    pub steps: u32,
}
impl TransmittanceTable {
    fn compute_parameters(size: [u16; 3], u_r: f64, u_µ: f64) -> (f64, f64) {
        assert!(u_r >= 0.0 && u_r <= 1.0);
        assert!(u_µ >= 0.0 && u_µ <= 1.0);

        let H = f64::sqrt(Rt * Rt - Rg * Rg);
        let ρ = u_r * H;
        let r = f64::sqrt(ρ * ρ + Rg * Rg);

        let hp = (size[1] / 2 - 1) as f64 / (size[1] - 1) as f64;
        let µ_horizon = -f64::sqrt(r * r - Rg * Rg) / r;
        let µ = if u_µ > 0.5 {
            let uu = (u_µ - (1.0 - hp)) / hp;
            f64::powf(uu, 5.0) * (1.0 - µ_horizon) + µ_horizon
        } else {
            let uu = u_µ / hp;
            -f64::powf(uu, 5.0) * (1.0 + µ_horizon) + µ_horizon
        };

        assert!(r >= Rg && r <= Rt && !r.is_nan());
        assert!(µ >= -1.0 && µ <= 1.0 && !µ.is_nan(), "{} {}", µ, u_µ);

        (r, µ)
    }
    fn reverse_parameters(size: [u16; 3], r: f64, µ: f64) -> (f64, f64) {
        assert!(r >= Rg && r <= Rt);
        assert!(µ >= -1.0 && µ <= 1.0);

        let H = f64::sqrt(Rt * Rt - Rg * Rg);
        let ρ = f64::sqrt(r * r - Rg * Rg);
        let u_r = ρ / H;

        let hp = (size[1] / 2 - 1) as f64 / (size[1] - 1) as f64;
        let µ_horizon = -f64::sqrt(r * r - Rg * Rg) / r;
        let u_µ = if µ > µ_horizon {
            let uu = f64::powf((µ - µ_horizon) / (1.0 - µ_horizon), 0.2);
            uu * hp + (1.0 - hp)
        } else {
            let uu = f64::powf((µ_horizon - µ) / (1.0 + µ_horizon), 0.2);
            uu * hp
        };

        assert!(u_r >= 0.0 && u_r <= 1.0 && !u_r.is_nan());
        assert!(u_µ >= 0.0 && u_µ <= 1.0 && !u_µ.is_nan());

        (u_r, u_µ)
    }
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
    fn size(&self) -> [u16; 3] {
        [1024, 1024, 1]
    }
    fn compute(&self, [x, y, _]: [u16; 3]) -> [f32; 4] {
        let (r, v) = Self::compute_parameters(
            self.size(),
            f64::from(x) / f64::from(self.size()[0] - 1),
            f64::from(y) / f64::from(self.size()[1] - 1),
        );

        assert!(v >= -1.0 && v <= 1.0, "AA {}", v);

        let intersects_ground = y < self.size()[1] / 2;
        let t = integral(r, f64::acos(v), self.steps, intersects_ground, |y| {
            let height = y.magnitude() - Rg;
            let βe_R = rayleigh::βe * f64::exp(-height / rayleigh::H);
            let βe_M = mie::βe * f64::exp(-height / mie::H);
            assert!(!βe_R.x.is_nan(), "{} {} {:?}", βe_R.x, height, y);
            assert!(!βe_M.is_nan());
            βe_R + Vector3::new(βe_M, βe_M, βe_M)
        });

        assert!(!t.x.is_nan());
        assert!(!t.y.is_nan());
        assert!(!t.z.is_nan());

        assert!(!f64::exp(-t.x).is_nan());
        assert!(!f64::exp(-t.y).is_nan());
        assert!(!f64::exp(-t.z).is_nan());

        assert!(!(f64::exp(-t.x) as f32).is_nan());
        assert!(!(f64::exp(-t.y) as f32).is_nan());
        assert!(!(f64::exp(-t.z) as f32).is_nan());

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
    fn compute_parameters(size: [u16; 3], u_r: f64, u_µ: f64, u_µ_s: f64) -> (f64, f64, f64) {
        assert!(u_r >= 0.0 && u_r <= 1.0);
        assert!(u_µ >= 0.0 && u_µ <= 1.0);
        assert!(u_µ_s >= 0.0 && u_µ_s <= 1.0);

        let H = f64::sqrt(Rt * Rt - Rg * Rg);
        let ρ = u_r * H;
        let r = f64::sqrt(ρ * ρ + Rg * Rg);

        let hp = (size[1] / 2 - 1) as f64 / (size[1] - 1) as f64;
        let µ_horizon = -f64::sqrt(r * r - Rg * Rg) / r;
        let µ = if u_µ > 0.5 {
            let uu = (u_µ - (1.0 - hp)) / hp;
            f64::powf(uu, 5.0) * (1.0 - µ_horizon) + µ_horizon
        } else {
            let uu = u_µ / hp;
            -f64::powf(uu, 5.0) * (1.0 + µ_horizon) + µ_horizon
        };

        let µ_s = (f64::tan((2.0 * u_µ_s - 1.0 + 0.26) * 0.75) / f64::tan(1.26 * 0.75))
            .max(-1.0)
            .min(1.0);

        (r, µ, µ_s)
    }
    #[cfg(test)]
    fn reverse_parameters(size: [u16; 3], r: f64, µ: f64, µ_s: f64) -> (f64, f64, f64) {
        assert!(r >= Rg && r <= Rt);
        assert!(µ >= -1.0 && µ <= 1.0);
        assert!(µ_s >= -1.0 && µ_s <= 1.0);

        let H = f64::sqrt(Rt * Rt - Rg * Rg);
        let ρ = f64::sqrt(r * r - Rg * Rg);
        let u_r = ρ / H;

        let hp = (size[1] / 2 - 1) as f64 / (size[1] - 1) as f64;
        let µ_horizon = -f64::sqrt(r * r - Rg * Rg) / r;
        let u_µ = if µ > µ_horizon {
            let uu = f64::powf((µ - µ_horizon) / (1.0 - µ_horizon), 0.2);
            uu * hp + (1.0 - hp)
        } else {
            let uu = f64::powf((µ_horizon - µ) / (1.0 + µ_horizon), 0.2);
            uu * hp
        };

        let u_µ_s =
            0.5 * (f64::atan(µ_s.max(-0.45) * f64::tan(1.26 * 0.75)) / 0.75 + (1.0 - 0.26));

        (u_r, u_µ, u_µ_s)
    }
}
impl LookupTableDefinition for InscatteringTable {
    fn filename(&self) -> String {
        format!(
            "sky/inscattering.{}x{}x{}.{:02}.bin",
            self.size()[0],
            self.size()[1],
            self.size()[2],
            self.steps
        )
    }
    fn size(&self) -> [u16; 3] {
        [1024, 1024, 32]
    }
    fn compute(&self, [x, y, z]: [u16; 3]) -> [f32; 4] {
        let (r, µ, µ_s) = Self::compute_parameters(
            self.size(),
            f64::from(x) / f64::from(self.size()[0] - 1),
            f64::from(y) / f64::from(self.size()[1] - 1),
            f64::from(z) / f64::from(self.size()[2] - 1),
        );

        let intersects_ground = y < self.size()[1] / 2;

        let (xx0, yy0) =
            TransmittanceTable::reverse_parameters(self.transmittance.size.clone(), r, µ);
        let [Tr0, Tg0, Tb0, _] = { self.transmittance.get2(xx0, yy0) };

        // let vv = if µ > 0.0 {
        //     Vector2::new(f64::sqrt(1.0 - µ * µ), µ)
        // } else {
        //     Vector2::new(-f64::sqrt(1.0 - µ * µ), -µ)
        // };
        let vv = Vector2::new(f64::sqrt(1.0 - µ * µ), µ);

        let L_sun = 1.0;
        let s = integral(r, f64::acos(µ), self.steps, intersects_ground, |y| {
            let y_magnitude = y.magnitude();
            let r = (y_magnitude).max(Rg);
            let h = r - Rg;

            let (xx, yy) =
                TransmittanceTable::reverse_parameters(self.transmittance.size.clone(), r, µ_s);
            let [Tr, Tg, Tb, _] = self.transmittance.get2(xx, yy);

            let (xx, yy) = TransmittanceTable::reverse_parameters(
                self.transmittance.size.clone(),
                r,
                y.dot(vv) / y_magnitude,
            );
            // if (yy > 0.5) != (yy0 > 0.5) {
            //     yy = yy0
            // }
            let [Tr1, Tg1, Tb1, _] = self.transmittance.get2(xx, yy);

            let Tr1 = Tr1.max(Tr0);
            let Tb1 = Tb1.max(Tb0);
            let Tg1 = Tg1.max(Tg0);

            let T = //if µ > 0.0 {
                Vector3::new(
                    (Tr * Tr0 / Tr1) as f64,
                    (Tg * Tg0 / Tg1) as f64,
                    (Tb * Tb0 / Tb1) as f64,
                );
            // } else {
            // Vector3::new(
            //     (Tr * Tr1 / Tr0) as f64,
            //     (Tg * Tg1 / Tg0) as f64,
            //     (Tb * Tb1 / Tb0) as f64,
            // );
            // };
            assert!(!T.x.is_nan() && !T.y.is_nan() && !T.z.is_nan());
            assert!(T.x >= 0. && T.y >= 0. && T.z >= 0.);
            assert!(T.x <= 1. && T.y <= 1. && T.z <= 1., "{} {} {}", µ, yy, yy0);

            let R = T.mul_element_wise(rayleigh::βs) * f64::exp(-h / rayleigh::H) * L_sun;
            let M = T.x * mie::βs * f64::exp(-h / mie::H) * L_sun * rayleigh::βs.x;
            Vector4::new(R.x, R.y, R.z, M)
        });
        [s.x as f32, s.y as f32, s.z as f32, s.w as f32]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cache::{AssetLoadContext, GeneratedAsset};
    use rand::{self, Rng};

    #[test]
    fn invert_transmittance_parameters() {
        let mut rng = rand::thread_rng();
        let size = [256, 1024, 1];
        for _ in 0..10000 {
            let (r, µ) = (rng.gen_range(Rg, Rt), rng.gen_range(-1.0, 1.0));

            let (x, y) = TransmittanceTable::reverse_parameters(size.clone(), r, µ);
            let (r2, µ2) = TransmittanceTable::compute_parameters(size.clone(), x, y);

            assert_relative_eq!(r, r2, max_relative = 0.0001);
            assert_relative_eq!(µ, µ2, max_relative = 0.0001);
        }
    }

    #[ignore]
    #[test]
    fn invert_inscatter_parameters() {
        let mut rng = rand::thread_rng();
        let size = [32, 256, 32];
        for _ in 0..1000 {
            let (x, y, z) = (
                rng.gen_range(0.0, 1.0),
                rng.gen_range(0.0, 1.0),
                rng.gen_range(0.0, 1.0),
            );

            let (r, µ, µ_s) = InscatteringTable::compute_parameters(size.clone(), x, y, z);
            let (x2, y2, z2) = InscatteringTable::reverse_parameters(size.clone(), r, µ, µ_s);

            assert_relative_eq!(x, x2, max_relative = 0.0001);
            assert_relative_eq!(y, y2, max_relative = 0.0001);
            assert_relative_eq!(z, z2, max_relative = 0.0001);
        }
    }

    #[test]
    #[ignore]
    fn transmittance_enough_steps() {
        let t1 = TransmittanceTable { steps: 1000 };
        let t2 = TransmittanceTable { steps: 2000 };

        let mut context = AssetLoadContext::new();
        let t1 = t1.load(&mut context).unwrap();
        let t2 = t2.load(&mut context).unwrap();
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