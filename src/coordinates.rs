
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

const WGS84_INV_FLATTENING: f64 = 298.257223563;
const WGS84_SEMI_MAJOR_AXIS_METERS: f64 = 6378137.0;
const WSG84_SEMI_MINOR_AXIS_METERS: f64 = WGS84_SEMI_MAJOR_AXIS_METERS * (1.0 - 1.0 / WGS84_INV_FLATTENING);

pub const PLANET_RADIUS: f64 = 6371000.0;

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
        ecef.x * PLANET_RADIUS / WGS84_SEMI_MAJOR_AXIS_METERS,
        ecef.y * PLANET_RADIUS / WGS84_SEMI_MAJOR_AXIS_METERS,
        ecef.z * PLANET_RADIUS / WSG84_SEMI_MINOR_AXIS_METERS,
    )
}

#[inline]
#[allow(unused)]
pub fn warped_to_ecef(warped: Vector3<f64>) -> Vector3<f64> {
    Vector3::new(
        warped.x * WGS84_SEMI_MAJOR_AXIS_METERS / PLANET_RADIUS,
        warped.y * WGS84_SEMI_MAJOR_AXIS_METERS / PLANET_RADIUS,
        warped.z * WSG84_SEMI_MINOR_AXIS_METERS / PLANET_RADIUS,
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
