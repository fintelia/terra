//! Astronomy functions
//!
//! The methods in this module are derived from the astro crate.
//! The original crate is available at https://github.com/saurvs/astro-rust
//!
//! ORIGINAL LICENSE NOTICE:
//!
//!    Copyright (c) 2015, 2016 Saurav Sachidanand
//!
//!    Permission is hereby granted, free of charge, to any person obtaining a copy of this
//!    software and associated documentation files (the "Software"), to deal in the Software
//!    without restriction, including without limitation the rights to use, copy, modify, merge,
//!    publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
//!    to whom the Software is furnished to do so, subject to the following conditions:
//!
//!    The above copyright notice and this permission notice shall be included in all copies or
//!    substantial portions of the Software.
//!
//!    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
//!    INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
//!    PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
//!    FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
//!    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//!    DEALINGS IN THE SOFTWARE.

/// Computes Julian century for a Julian day
///
/// # Arguments
///
/// * `JD`: Julian (Ephemeris) day
#[inline]
pub(crate) fn julian_cent(jd: f64) -> f64 {
    (jd - 2451545.0) / 36525.0
}

/// Computes mean sidereal time for a Julian day
///
/// # Returns
///
/// * `mn_sidr`: Mean sidereal time *| in radians*
///
/// # Arguments
///
/// * `JD`: Julian day
pub(crate) fn mn_sidr(jd: f64) -> f64 {
    let jc = julian_cent(jd);

    limit_to_360(
        280.46061837
            + 360.98564736629 * (jd - 2451545.0)
            + jc * jc * (0.000387933 - jc / 38710000.0),
    )
    .to_radians()
}

/// Computes the equivalent angle in [0, 360] degree range
///
/// # Arguments
///
/// * `angl`: Angle *| in degrees*
#[inline]
fn limit_to_360(angl: f64) -> f64 {
    let n = (angl / 360.0) as i64;
    let limited_angl = angl - (360.0 * (n as f64));

    if limited_angl < 0.0 {
        limited_angl + 360.0
    } else {
        limited_angl
    }
}

/// Computes the right ascension from ecliptic coordinates
///
/// # Returns
///
/// * `asc`: Right ascension *| in radians*
///
/// # Arguments
///
/// * `ecl_long`: Ecliptic longitude *| in radians*
/// * `ecl_lat`: Ecliptic latitude *| in radians*
/// * `oblq_eclip`: If `ecl_long` and `ecl_lat` are corrected
///                     for nutation, then *true* obliquity. If not, then
///                     *mean* obliquity. *| in radians*
pub(crate) fn asc_frm_ecl(ecl_long: f64, ecl_lat: f64, oblq_eclip: f64) -> f64 {
    (ecl_long.sin() * oblq_eclip.cos() - ecl_lat.tan() * oblq_eclip.sin()).atan2(ecl_long.cos())
}

/// Computes the declination from ecliptic coordinates
///
/// # Returns
///
/// * `dec`: Declination *| in radians*
///
/// # Arguments
///
/// * `ecl_long`: Ecliptic longitude *| in radians*
/// * `ecl_lat`: Ecliptic latitude *| in radians*
/// * `oblq_eclip`: If `ecl_long` and `ecl_lat` are corrected
///                     for nutation, then *true* obliquity. If not, then
///                     *mean* obliquity. *| in radians*
pub(crate) fn dec_frm_ecl(ecl_long: f64, ecl_lat: f64, oblq_eclip: f64) -> f64 {
    (ecl_lat.sin() * oblq_eclip.cos() + ecl_lat.cos() * oblq_eclip.sin() * ecl_long.sin()).asin()
}

/// Computes the right ascension from galactic coordinates
///
/// # Returns
///
/// * `asc`: Right ascension *| in radians*
///
/// The right ascension returned here is referred to the standard equinox
/// of  B1950.0.
///
/// # Arguments
///
/// * `gal_long`: Galactic longitude *| in radians*
/// * `gal_lat`: Galactic latitude *| in radians*
pub(crate) fn asc_frm_gal(gal_long: f64, gal_lat: f64) -> f64 {
    12.25_f64.to_radians()
        + (gal_long - 123_f64.to_radians()).sin().atan2(
            27.4_f64.to_radians().sin() * (gal_long - 123_f64.to_radians()).cos()
                - 27.4_f64.to_radians().cos() * gal_lat.tan(),
        )
}

/// Computes the declination from galactic coordinates
///
/// # Returns
///
/// * `dec`: Declination *| in radians*
///
/// The declination returned here is referred to the standard equinox
/// of  B1950.0.
///
/// # Arguments
///
/// * `gal_long`: Galactic longitude *| in radians*
/// * `gal_lat`: Galactic latitude *| in radians*
pub(crate) fn dec_frm_gal(gal_long: f64, gal_lat: f64) -> f64 {
    (gal_lat.sin() * 27.4_f64.to_radians().sin()
        + gal_lat.cos() * 27.4_f64.to_radians().cos() * (gal_long - 123_f64.to_radians()).cos())
    .asin()
}
