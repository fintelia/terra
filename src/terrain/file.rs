use std::f64::consts::PI;

use terrain::dem;

/// This file assumes that all coordinates are provided relative to the earth represented as a
/// perfect sphere. This isn't quite accurate: The coordinate system of the input datasets are
/// actually WGS84 or NAD83. However, for our purposes the difference should not be noticable.

/// The radius of the earth in meters.
const EARTH_RADIUS: f64 = 6371000.0;
const EARTH_CIRCUMFERENCE: f64 = 2.0 * PI * EARTH_RADIUS;

pub struct TerrainFile {
    width: usize,
    height: usize,
    cell_size: f32,

    elevations: Vec<f32>,
    slopes: Vec<(f32, f32)>,
}

impl TerrainFile {
    fn compute_slopes(width: usize,
                      height: usize,
                      cell_size: f32,
                      elevations: &[f32])
                      -> Vec<(f32, f32)> {
        let get_elevation = |x: usize, y: usize| {
            assert!(x < width);
            assert!(y < height);
            elevations[x + y * width]
        };

        let mut slopes = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let xslope = if x == 0 {
                    (get_elevation(x + 1, y) - get_elevation(x, y)) / cell_size
                } else if x == width - 1 {
                    (get_elevation(x, y) - get_elevation(x - 1, y)) / cell_size
                } else {
                    (get_elevation(x + 1, y) - get_elevation(x - 1, y)) / (2.0 * cell_size)
                };

                let yslope = if y == 0 {
                    (get_elevation(x, y + 1) - get_elevation(x, y)) / cell_size
                } else if y == height - 1 {
                    (get_elevation(x, y) - get_elevation(x, y - 1)) / cell_size
                } else {
                    (get_elevation(x, y + 1) - get_elevation(x, y - 1)) / (2.0 * cell_size)
                };

                slopes.push((xslope, yslope));
            }
        }

        slopes
    }

    /// Construct a `TerrainFile` from a `DigitalElevationModel`.
    pub fn from_digital_elevation_model(dem: dem::DigitalElevationModel) -> Self {
        // Compute approximate cell size in meters.
        let xcenter = dem.xllcorner + dem.cell_size * dem.width as f64;
        let ycenter = dem.yllcorner + dem.cell_size * dem.height as f64;
        let cell_size_x = (dem.cell_size / 360.0) * EARTH_CIRCUMFERENCE *
                          ycenter.to_radians().cos();
        let cell_size_y = (dem.cell_size / 360.0) * EARTH_CIRCUMFERENCE;
        let cell_size = 0.5 * (cell_size_x + cell_size_y) as f32;

        let slopes = Self::compute_slopes(dem.width, dem.height, cell_size, &dem.elevations[..]);

        TerrainFile {
            width: dem.width,
            height: dem.height,
            cell_size,
            elevations: dem.elevations,
            slopes,
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.width
    }

    pub fn elevation(&self, x: usize, y: usize) -> f32 {
        assert!(x < self.width);
        assert!(y < self.height);
        self.elevations[x + y * self.width]
    }

    pub fn slope(&self, x: usize, y: usize) -> (f32, f32) {
        assert!(x < self.width);
        assert!(y < self.height);
        self.slopes[x + y * self.width]
    }

    pub fn elevations(&self) -> &[f32] {
        &self.elevations[..]
    }
}
