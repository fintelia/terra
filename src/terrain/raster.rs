use bit_vec::BitVec;
use lru_cache::LruCache;

use cache::AssetLoadContext;
use coordinates;

use std::collections::HashSet;
use std::f64::consts::PI;
use std::ops::Index;

/// Wrapper around BitVec that converts values to f64's so that it can be uses as a backing for a
/// GlobalRaster.
pub struct BitContainer(pub BitVec<u32>);
impl Index<usize> for BitContainer {
    type Output = f64;
    fn index(&self, i: usize) -> &f64 {
        if self.0[i] {
            &255.0
        } else {
            &0.0
        }
    }
}

/// Currently assumes that values are taken at the lower left corner of each cell.
#[derive(Serialize, Deserialize)]
pub struct Raster<T: Into<f64> + Copy> {
    pub width: usize,
    pub height: usize,
    pub cell_size: f64,

    pub xllcorner: f64,
    pub yllcorner: f64,

    pub values: Vec<T>,
}

impl<T: Into<f64> + Copy> Raster<T> {
    pub fn interpolate(&self, latitude: f64, longitude: f64) -> Option<f64> {
        let x = (latitude - self.xllcorner) / self.cell_size;
        let y = (longitude - self.yllcorner) / self.cell_size;

        let y = (self.height - 1) as f64 - y;

        let fx = x.floor() as usize;
        let fy = y.floor() as usize;
        if x < 0.0 || fx >= self.width - 1 || y < 0.0 || fy >= self.height - 1 {
            return None;
        }

        let h00 = self.values[fx + fy * self.width].into();
        let h10 = self.values[fx + 1 + fy * self.width].into();
        let h01 = self.values[fx + (fy + 1) * self.width].into();
        let h11 = self.values[fx + 1 + (fy + 1) * self.width].into();
        let h0 = h00 + (h01 - h00) * (y - fy as f64);
        let h1 = h10 + (h11 - h10) * (y - fy as f64);
        Some(h0 + (h1 - h0) * (x - fx as f64))
    }
}

pub(crate) trait RasterSource {
    type Type: Into<f64> + Copy;
    fn load(
        &self,
        context: &mut AssetLoadContext,
        latitude: i16,
        longitude: i16,
    ) -> Option<Raster<Self::Type>>;
}

pub(crate) struct RasterCache<T: Into<f64> + Copy> {
    source: Box<RasterSource<Type = T>>,
    holes: HashSet<(i16, i16)>,
    rasters: LruCache<(i16, i16), Raster<T>>,
}
impl<T: Into<f64> + Copy> RasterCache<T> {
    pub fn new(source: Box<RasterSource<Type = T>>, size: usize) -> Self {
        Self {
            source,
            holes: HashSet::new(),
            rasters: LruCache::new(size),
        }
    }
    pub fn interpolate(
        &mut self,
        context: &mut AssetLoadContext,
        latitude: f64,
        longitude: f64,
    ) -> Option<f64> {
        let key = (latitude.floor() as i16, longitude.floor() as i16);

        if self.holes.contains(&key) {
            return None;
        }
        if let Some(dem) = self.rasters.get_mut(&key) {
            return dem.interpolate(latitude, longitude);
        }

        match self.source.load(context, key.0, key.1) {
            Some(raster) => {
                let value = raster.interpolate(latitude, longitude);
                // assert!(value.is_some());
                self.rasters.insert(key, raster);
                value
            }
            None => {
                self.holes.insert(key);
                None
            }
        }
    }
}

/// Currently assumes that values are taken at the *center* of cells.
pub(crate) struct GlobalRaster<T: Into<f64> + Copy, C: Index<usize, Output = T> = Vec<T>> {
    pub width: usize,
    pub height: usize,
    pub bands: usize,
    pub values: C,
}
impl<T: Into<f64> + Copy, C: Index<usize, Output = T>> GlobalRaster<T, C> {
    /// Returns the approximate grid spacing in meters.
    pub fn spacing(&self) -> f64 {
        let sx = 2.0 * PI * coordinates::PLANET_RADIUS / self.width as f64;
        let sy = PI * coordinates::PLANET_RADIUS / self.height as f64;
        sx.min(sy)
    }

    fn get(&self, x: i64, y: i64, band: usize) -> f64 {
        let y = y.max(0).min(self.height as i64) as usize;
        let x = (((x % self.width as i64) + self.width as i64) % self.width as i64) as usize;
        self.values[(x + y * self.width) * self.bands + band].into()
    }

    pub fn interpolate(&self, latitude: f64, longitude: f64, band: usize) -> f64 {
        assert!(latitude >= -90.0 && latitude <= 90.0);

        let x = (longitude + 180.0) / 360.0 * self.width as f64 - 0.5;
        let y = (90.0 - latitude) / 180.0 * self.height as f64 - 0.5;

        let fx = x.floor() as i64;
        let fy = y.floor() as i64;

        let h00 = self.get(fx, fy, band);
        let h10 = self.get(fx + 1, fy, band);
        let h01 = self.get(fx, fy + 1, band);
        let h11 = self.get(fx + 1, fy + 1, band);
        let h0 = h00 + (h01 - h00) * (y - fy as f64);
        let h1 = h10 + (h11 - h10) * (y - fy as f64);
        h0 + (h1 - h0) * (x - fx as f64)
    }
}
