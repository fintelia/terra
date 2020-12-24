
//! This module uses a number of different coordinate systems and provides conversions between them.
//!
//! *ecef* - Cartesian coordinate system centered at the planet center, and also with units of
//! meters. The x-axis points to 0°N 0°E, the y-axis points towards 0°N 90°E, and the z-axis points
//! towards the north pole. Commonly referred to as "earth-centered, earth-fixed".
//!
//! *warped* - Coordinate system centered at the planet center, but warped such that all points on
//! the planet surface are distance PLANET_RADIUS from the origin. Useful for sky rendering because
//! the ellipsoidal shape of the earth can be ignored.
//!
//! *lla* - Consist of latitude, longitude, and altitude (above sea level). Angle measurements
//! are given in radians, and altitude in meters.
//!
//! *polar* - Same as lla, but assumes a perfectly spherical planet which makes conversions
//! considerably faster.
//!
//! *cspace* - Restricted to points on the unit cube, projected from polar.

use cgmath::{InnerSpace, Vector3};
use coord_transforms::geo;
use coord_transforms::structs::geo_ellipsoid::geo_ellipsoid as GeoEllipsoid;
use coord_transforms::structs::geo_ellipsoid::{WGS84_SEMI_MAJOR_AXIS_METERS, WGS84_FLATTENING};
use nalgebra as na;

lazy_static! {
    static ref ELLIPSOID: GeoEllipsoid =
        GeoEllipsoid::new(WGS84_SEMI_MAJOR_AXIS_METERS, WGS84_FLATTENING);
}

pub const PLANET_RADIUS: f64 = 6371000.0;


#[allow(unused)]
pub fn ecef_to_lla(ecef: Vector3<f64>) -> Vector3<f64> {
    let lla = geo::ecef2lla(&na::Vector3::new(ecef.x, ecef.y, ecef.z), &ELLIPSOID);
    Vector3::new(lla.x, lla.y, lla.z)
}
#[allow(unused)]
pub fn lla_to_ecef(lla: Vector3<f64>) -> Vector3<f64> {
    let ecef = geo::lla2ecef(&na::Vector3::new(lla.x, lla.y, lla.z), &ELLIPSOID);
    Vector3::new(ecef.x, ecef.y, ecef.z)
}

#[inline]
#[allow(unused)]
pub fn ecef_to_polar(ecef: Vector3<f64>) -> Vector3<f64> {
    let r = f64::sqrt(ecef.x * ecef.x + ecef.y * ecef.y + ecef.z * ecef.z);
    Vector3::new(f64::asin(ecef.z / r), f64::atan2(ecef.y, ecef.x), r - PLANET_RADIUS)
}
#[inline]
#[allow(unused)]
pub fn polar_to_ecef(lla: Vector3<f64>) -> Vector3<f64> {
    Vector3::new(
        (PLANET_RADIUS + lla.z) * f64::cos(lla.x) * f64::cos(lla.y),
        (PLANET_RADIUS + lla.z) * f64::cos(lla.x) * f64::sin(lla.y),
        (PLANET_RADIUS + lla.z) * f64::sin(lla.x),
    )
}

#[inline]
#[allow(unused)]
pub fn ecef_to_warped(ecef: Vector3<f64>) -> Vector3<f64> {
    Vector3::new(
        ecef.x * PLANET_RADIUS / ELLIPSOID.get_semi_major_axis(),
        ecef.y * PLANET_RADIUS / ELLIPSOID.get_semi_major_axis(),
        ecef.z * PLANET_RADIUS / ELLIPSOID.get_semi_minor_axis(),
    )
}

#[inline]
#[allow(unused)]
pub fn warped_to_ecef(warped: Vector3<f64>) -> Vector3<f64> {
    Vector3::new(
        warped.x * ELLIPSOID.get_semi_major_axis() / PLANET_RADIUS,
        warped.y * ELLIPSOID.get_semi_major_axis() / PLANET_RADIUS,
        warped.z * ELLIPSOID.get_semi_minor_axis() / PLANET_RADIUS,
    )
}

#[allow(unused)]
pub fn sun_direction() -> Vector3<f64> {
    use astro::{coords, sun};

    let (ecl, distance_au) = sun::geocent_ecl_pos(180.0);
    let distance = distance_au * 149597870700.0;

    let e = 0.40905;
    let declination = coords::dec_frm_ecl(ecl.long, ecl.lat, e);
    let right_ascension = coords::asc_frm_ecl(ecl.long, ecl.lat, e);

    let eq_rect = Vector3::new(
        distance * declination.cos() * right_ascension.cos(),
        distance * declination.cos() * right_ascension.sin(),
        distance * declination.sin(),
    );

    // TODO: Is this conversion from equatorial coordinates to ECEF actually valid?
    let ecef = Vector3::new(eq_rect.x, -eq_rect.y, eq_rect.z);

    ecef.normalize()
}

pub fn cspace_to_polar(position: Vector3<f64>) -> Vector3<f64> {
    let p = Vector3::new(position.x, position.y, position.z).normalize();
    let latitude = f64::asin(p.z);
    let longitude = f64::atan2(p.y, p.x);
    Vector3::new(latitude, longitude, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::assert_relative_eq;

    fn vec3_na2cgmath(v: na::Vector3<f64>) -> Vector3<f64> {
        Vector3::new(v.x, v.y, v.z)
    }

    #[test]
    fn test_lla_to_ecef() {
        assert_eq!(
            lla_to_ecef(Vector3::new(0.2345, -0.637, 10.0)),
            vec3_na2cgmath(geo::lla2ecef(&na::Vector3::new(0.2345, -0.637, 10.0), &ELLIPSOID)),
        );
        assert_relative_eq!(
            lla_to_ecef(Vector3::new(19.0f64.to_radians(), -34.0f64.to_radians(), 10.0,)),
            Vector3::new(5001415.53283897, -3373497.37316789, 2063352.71871789),
            epsilon = 0.001
        );
    }

    #[test]
    fn test_lla_ecef_lla() {
        let roundtrip =
            |p: Vector3<f64>| -> Vector3<f64> { ecef_to_lla(lla_to_ecef(p)) };

        let a = Vector3::new(0.0, 0.0, 50.0);
        let b = Vector3::new(40.0f64.to_radians(), 50.0f64.to_radians(), 100.0);

        assert_relative_eq!(a, roundtrip(a), epsilon = 0.1);
        assert_relative_eq!(b, roundtrip(b), epsilon = 0.1);
    }

    #[test]
    fn test_ecef_lla_ecef() {
        let roundtrip =
            |p: Vector3<f64>| -> Vector3<f64> { lla_to_ecef(ecef_to_lla(p)) };

        let a = Vector3::new(-5280434.995591136, 342.0201433256688, 4429584.375659218);
        let b = Vector3::new(-4880469.147111009, 0.0, 4095199.8613129416);

        assert_relative_eq!(a, roundtrip(a), epsilon = 0.1);
        assert_relative_eq!(b, roundtrip(b), epsilon = 0.1);
    }
}
