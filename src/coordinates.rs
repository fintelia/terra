use cgmath::{Matrix, Matrix3, Matrix4, Vector2, Vector3};
use coord_transforms::geo;
use coord_transforms::structs::geo_ellipsoid::geo_ellipsoid as GeoEllipsoid;
use coord_transforms::structs::geo_ellipsoid::*;
use nalgebra as na;
use serde::{Deserialize, Serialize};

lazy_static! {
    static ref ELLIPSOID: GeoEllipsoid =
        { GeoEllipsoid::new(WGS84_SEMI_MAJOR_AXIS_METERS, WGS84_FLATTENING) };
}

pub const PLANET_RADIUS: f64 = 6371000.0;

/// This struct uses a number of different coordinate systems and provides conversions between them.
///
/// *world* - Cartesian coordinate system with units of meters and centered at a point on the planet
/// surface. The x-axis points east, the y-axis points up, and the z-axis points south.
///
/// *ecef* - Cartesian coordinate system centered at the planet center, and also with units of
/// meters. The x-axis points to 0°N 0°E, the y-axis points towards 0°N 90°E, and the z-axis points
/// towards the north pole. Commonly referred to as "earth-centered, earth-fixed".
///
/// *warped* - Coordinate system centered at the planet center, but warped such that all points on
/// the planet surface are distance PLANET_RADIUS from the origin. Useful for sky rendering because
/// the ellipsoidal shape of the earth can be ignored.
///
/// *lla* - Consist of latitude, longitude, and altitude (above sea level). Angle measurements
/// are given in radians, and altitude in meters.
///
/// *polar* - Same as lla, but assumes a perfectly spherical planet which makes conversions
/// considerably faster.
#[derive(Debug, Serialize, Deserialize)]
pub struct CoordinateSystem {
    center_ecef: Vector3<f64>,
    ecef_to_ned_matrix: Matrix3<f64>,
}

#[allow(unused)]
impl CoordinateSystem {
    fn make_ecef_to_ned_matrix(center_lla: Vector3<f64>) -> Matrix3<f64> {
        let sin_lat = center_lla.x.sin();
        let cos_lat = center_lla.x.cos();
        let sin_long = center_lla.y.sin();
        let cos_long = center_lla.y.cos();
        Matrix3::new(
            -sin_lat * cos_long,
            -sin_lat * sin_long,
            cos_lat,
            -sin_long,
            cos_long,
            0.0,
            -cos_lat * cos_long,
            -cos_lat * sin_long,
            -sin_lat,
        ).transpose()
    }

    pub fn from_lla(center_lla: Vector3<f64>) -> Self {
        let ecef_to_ned_matrix = Self::make_ecef_to_ned_matrix(center_lla);
        let center_ecef = geo::lla2ecef(
            &na::Vector3::new(center_lla.x, center_lla.y, center_lla.z),
            &ELLIPSOID,
        );

        Self {
            ecef_to_ned_matrix,
            center_ecef: Vector3::new(center_ecef.x, center_ecef.y, center_ecef.z),
        }
    }

    pub fn ecef_to_lla(&self, ecef: Vector3<f64>) -> Vector3<f64> {
        let lla = geo::ecef2lla(&na::Vector3::new(ecef.x, ecef.y, ecef.z), &ELLIPSOID);
        Vector3::new(lla.x, lla.y, lla.z)
    }
    pub fn lla_to_ecef(&self, lla: Vector3<f64>) -> Vector3<f64> {
        let ecef = geo::lla2ecef(&na::Vector3::new(lla.x, lla.y, lla.z), &ELLIPSOID);
        Vector3::new(ecef.x, ecef.y, ecef.z)
    }

    #[inline]
    pub fn ecef_to_polar(&self, ecef: Vector3<f64>) -> Vector3<f64> {
        let r = f64::sqrt(ecef.x * ecef.x + ecef.y * ecef.y + ecef.z * ecef.z);
        Vector3::new(
            f64::asin(ecef.z / r),
            f64::atan2(ecef.y, ecef.x),
            r - PLANET_RADIUS,
        )
    }
    #[inline]
    pub fn polar_to_ecef(&self, lla: Vector3<f64>) -> Vector3<f64> {
        Vector3::new(
            (PLANET_RADIUS + lla.z) * f64::cos(lla.x) * f64::cos(lla.y),
            (PLANET_RADIUS + lla.z) * f64::cos(lla.x) * f64::sin(lla.y),
            (PLANET_RADIUS + lla.z) * f64::sin(lla.x),
        )
    }

    #[inline]
    pub fn ned_to_ecef(&self, ned: Vector3<f64>) -> Vector3<f64> {
        self.ecef_to_ned_matrix.transpose() * ned + self.center_ecef
    }
    #[inline]
    pub fn ecef_to_ned(&self, ecef: Vector3<f64>) -> Vector3<f64> {
        self.ecef_to_ned_matrix * (ecef - self.center_ecef)
    }

    #[inline]
    pub fn world_to_ned(&self, world: Vector3<f64>) -> Vector3<f64> {
        Vector3::new(-world.z, world.x, -world.y)
    }
    #[inline]
    pub fn ned_to_world(&self, ned: Vector3<f64>) -> Vector3<f64> {
        Vector3::new(ned.y, -ned.z, -ned.x)
    }

    #[inline]
    pub fn world_to_ecef(&self, world: Vector3<f64>) -> Vector3<f64> {
        self.ned_to_ecef(self.world_to_ned(world))
    }
    #[inline]
    pub fn world_to_lla(&self, world: Vector3<f64>) -> Vector3<f64> {
        self.ecef_to_lla(self.world_to_ecef(world))
    }
    #[inline]
    pub fn world_to_polar(&self, world: Vector3<f64>) -> Vector3<f64> {
        self.ecef_to_polar(self.world_to_ecef(world))
    }

    #[inline]
    pub fn ecef_to_world(&self, ecef: Vector3<f64>) -> Vector3<f64> {
        self.ned_to_world(self.ecef_to_ned(ecef))
    }
    #[inline]
    pub fn lla_to_world(&self, lla: Vector3<f64>) -> Vector3<f64> {
        self.ecef_to_world(self.lla_to_ecef(lla))
    }
    #[inline]
    pub fn polar_to_world(&self, lla: Vector3<f64>) -> Vector3<f64> {
        self.ecef_to_world(self.polar_to_ecef(lla))
    }

    #[inline]
    pub fn ecef_to_warped(&self, ecef: Vector3<f64>) -> Vector3<f64> {
        Vector3::new(
            ecef.x * PLANET_RADIUS / ELLIPSOID.get_semi_major_axis(),
            ecef.y * PLANET_RADIUS / ELLIPSOID.get_semi_major_axis(),
            ecef.z * PLANET_RADIUS / ELLIPSOID.get_semi_minor_axis(),
        )
    }

    #[inline]
    pub fn warped_to_ecef(&self, warped: Vector3<f64>) -> Vector3<f64> {
        Vector3::new(
            warped.x * ELLIPSOID.get_semi_major_axis() / PLANET_RADIUS,
            warped.y * ELLIPSOID.get_semi_major_axis() / PLANET_RADIUS,
            warped.z * ELLIPSOID.get_semi_minor_axis() / PLANET_RADIUS,
        )
    }

    #[inline]
    pub fn world_to_warped(&self, world: Vector3<f64>) -> Vector3<f64> {
        self.ecef_to_warped(self.world_to_ecef(world))
    }

    #[inline]
    pub fn warped_to_world(&self, warped: Vector3<f64>) -> Vector3<f64> {
        self.ecef_to_world(self.warped_to_ecef(warped))
    }

    pub fn world_to_warped_matrix(&self) -> Matrix4<f64> {
        let c = self.world_to_warped(Vector3::new(0.0, 0.0, 0.0));
        let x = self.world_to_warped(Vector3::new(1.0, 0.0, 0.0)) - c;
        let y = self.world_to_warped(Vector3::new(0.0, 1.0, 0.0)) - c;
        let z = self.world_to_warped(Vector3::new(0.0, 0.0, 1.0)) - c;

        #[rustfmt::skip]
        Matrix4::new(x.x, y.x, z.x, c.x,
                     x.y, y.y, z.y, c.y,
                     x.z, y.z, z.z, c.z,
                     0.0, 0.0, 0.0, 1.0).transpose()
    }

    pub fn world_height_on_surface(&self, world_xz: Vector2<f64>) -> f64 {
        let mut world3 = Vector3::new(0., 0., 0.);
        for i in 0..5 {
            world3.x = world_xz.x;
            world3.z = world_xz.y;
            let mut lla = self.world_to_lla(world3);
            lla.z = 0.0;
            world3 = self.lla_to_world(lla);
        }
        world3.y
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::{InnerSpace, Vector4};

    fn vec3_na2cgmath(v: na::Vector3<f64>) -> Vector3<f64> {
        Vector3::new(v.x, v.y, v.z)
    }

    #[test]
    fn lla_to_ecef() {
        let system =
            CoordinateSystem::from_lla(Vector3::new(40f64.to_radians(), 70f64.to_radians(), 0.0));
        assert_eq!(
            system.lla_to_ecef(Vector3::new(0.2345, -0.637, 10.0)),
            vec3_na2cgmath(geo::lla2ecef(
                &na::Vector3::new(0.2345, -0.637, 10.0),
                &ELLIPSOID
            )),
        );
        assert_relative_eq!(
            system.lla_to_ecef(Vector3::new(
                19.0f64.to_radians(),
                -34.0f64.to_radians(),
                10.0,
            )),
            Vector3::new(5001415.53283897, -3373497.37316789, 2063352.71871789),
            epsilon = 0.001
        );
    }

    #[test]
    fn ecef_to_ned() {
        let system =
            CoordinateSystem::from_lla(Vector3::new(40f64.to_radians(), 70f64.to_radians(), 0.0));
        assert_relative_eq!(
            system.ecef_to_ned(Vector3::new(1., 2., 3.)),
            Vector3::new(21054.5066838982, -0.255652334075421, 6369306.43707447),
            epsilon = 0.001
        );
        assert_relative_eq!(
            system.ecef_to_ned(Vector3::new(43., 32., -85.)),
            Vector3::new(20959.7405446614, -29.4621381075121, 6369330.40288434),
            epsilon = 0.001
        );
    }

    #[test]
    fn world_to_ned() {
        let system =
            CoordinateSystem::from_lla(Vector3::new(40f64.to_radians(), 70f64.to_radians(), 0.0));
        assert_eq!(
            system.world_to_ned(Vector3::new(0.0, 0.0, 0.0)),
            Vector3::new(0.0, 0.0, 0.0)
        );
        assert_relative_eq!(
            system.world_to_ned(Vector3::new(0.0, 0.0, -10.0)),
            Vector3::new(10.0, 0.0, 0.0)
        );
        assert_relative_eq!(
            system.world_to_ned(Vector3::new(10.0, 5.0, -10.0)),
            Vector3::new(10.0, 10.0, -5.0)
        );
    }

    #[test]
    fn lla_to_ned() {
        let lla_origin = na::Vector3::new(10.0f64.to_radians(), 87.0f64.to_radians(), 0.0);
        let system = CoordinateSystem::from_lla(vec3_na2cgmath(lla_origin));
        assert_eq!(
            system.ecef_to_ned(system.lla_to_ecef(Vector3::new(0.0, 0.0, 7.0))),
            vec3_na2cgmath(geo::lla2ned(
                &lla_origin,
                &na::Vector3::new(0.0, 0.0, 7.0),
                &ELLIPSOID
            )),
        );
        assert_eq!(
            system.ecef_to_ned(system.lla_to_ecef(Vector3::new(43.0, -450.0, 10.0))),
            vec3_na2cgmath(geo::lla2ned(
                &lla_origin,
                &na::Vector3::new(43.0, -450.0, 10.0),
                &ELLIPSOID
            )),
        );
        assert_eq!(
            system.ecef_to_ned(system.lla_to_ecef(Vector3::new(-865.0, 1.0, -9.0))),
            vec3_na2cgmath(geo::lla2ned(
                &lla_origin,
                &na::Vector3::new(-865.0, 1.0, -9.0),
                &ELLIPSOID
            )),
        );
    }

    #[test]
    fn world_to_lla() {
        let center_lla = Vector3::new(40f64.to_radians() as f64, 70f64.to_radians() as f64, 0.0);
        let system = CoordinateSystem::from_lla(center_lla);
        let p = system.world_to_lla(Vector3::new(0., 0., 0.));
        assert_relative_eq!(p, center_lla, epsilon = 0.001);
    }

    #[test]
    fn lla_ecef_lla() {
        let system =
            CoordinateSystem::from_lla(Vector3::new(40f64.to_radians(), 70f64.to_radians(), 0.0));
        let roundtrip =
            |p: Vector3<f64>| -> Vector3<f64> { system.ecef_to_lla(system.lla_to_ecef(p)) };

        let a = Vector3::new(0.0, 0.0, 50.0);
        let b = Vector3::new(40.0f64.to_radians(), 50.0f64.to_radians(), 100.0);

        assert_relative_eq!(a, roundtrip(a), epsilon = 0.1);
        assert_relative_eq!(b, roundtrip(b), epsilon = 0.1);
    }

    #[test]
    fn ecef_lla_ecef() {
        let system =
            CoordinateSystem::from_lla(Vector3::new(40f64.to_radians(), 70f64.to_radians(), 0.0));
        let roundtrip =
            |p: Vector3<f64>| -> Vector3<f64> { system.lla_to_ecef(system.ecef_to_lla(p)) };

        let a = Vector3::new(-5280434.995591136, 342.0201433256688, 4429584.375659218);
        let b = Vector3::new(-4880469.147111009, 0.0, 4095199.8613129416);

        assert_relative_eq!(a, roundtrip(a), epsilon = 0.1);
        assert_relative_eq!(b, roundtrip(b), epsilon = 0.1);
    }

    #[test]
    fn world_polar_world() {
        let system =
            CoordinateSystem::from_lla(Vector3::new(40f64.to_radians(), 70f64.to_radians(), 0.0));

        let roundtrip =
            |p: Vector3<f64>| -> Vector3<f64> { system.polar_to_world(system.world_to_polar(p)) };

        let a = Vector3::new(-52434.9, 342.0, 49584.3);
        let b = Vector3::new(-4469.1, 4356.0, 40999.8);

        assert_relative_eq!(a, roundtrip(a), epsilon = 0.1);
        assert_relative_eq!(b, roundtrip(b), epsilon = 0.1);
    }

    #[test]
    fn world_lla_world() {
        let system =
            CoordinateSystem::from_lla(Vector3::new(40f64.to_radians(), 70f64.to_radians(), 0.0));

        let roundtrip =
            |p: Vector3<f64>| -> Vector3<f64> { system.lla_to_world(system.world_to_lla(p)) };

        let a = Vector3::new(-52434.9, 342.0, 49584.3);
        let b = Vector3::new(-4469.1, 4356.0, 40999.8);

        assert_relative_eq!(a, roundtrip(a), epsilon = 0.1);
        assert_relative_eq!(b, roundtrip(b), epsilon = 0.1);
    }

    #[test]
    fn world_height_on_surface() {
        let system =
            CoordinateSystem::from_lla(Vector3::new(40f64.to_radians(), 70f64.to_radians(), 0.0));

        let a = Vector2::new(0.0, 0.0);
        let b = Vector2::new(4469.1, 40999.8);

        assert_relative_eq!(
            system
                .world_to_lla(Vector3::new(a.x, system.world_height_on_surface(a), a.y))
                .z,
            0.0
        );
        assert_relative_eq!(
            system
                .world_to_lla(Vector3::new(b.x, system.world_height_on_surface(b), b.y))
                .z,
            0.0
        );
    }

    #[test]
    fn world_to_warped() {
        let system =
            CoordinateSystem::from_lla(Vector3::new(40f64.to_radians(), 70f64.to_radians(), 0.0));
        let world = system.lla_to_world(Vector3::new(1.0, 0.5, 0.0));
        let warped = system.world_to_warped(world);
        assert_relative_eq!(warped.magnitude(), PLANET_RADIUS, epsilon = 0.01);
        assert_relative_eq!(world, system.warped_to_world(warped), epsilon = 0.01);
    }

    #[test]
    fn warped_to_world_matrix() {
        let system =
            CoordinateSystem::from_lla(Vector3::new(40f64.to_radians(), 70f64.to_radians(), 0.0));
        let matrix = system.world_to_warped_matrix();
        let a = Vector3::new(-534.9, 342.0, 584.3);
        let aw = matrix * Vector4::new(a.x, a.y, a.z, 1.0);
        let aw = Vector3::new(aw.x / aw.w, aw.y / aw.w, aw.z / aw.w);
        assert_relative_eq!(system.world_to_warped(a), aw, epsilon = 0.01);
    }
}
