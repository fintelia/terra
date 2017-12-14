use lru_cache::LruCache;

use cache::AssetLoadContext;

use std::collections::HashSet;

/// Currently assumes that values are taken at the lower left corner of each cell.
#[derive(Serialize, Deserialize)]
pub struct Raster {
    pub width: usize,
    pub height: usize,
    pub cell_size: f64,

    pub xllcorner: f64,
    pub yllcorner: f64,

    pub values: Vec<f32>,
}

impl Raster {
    pub fn interpolate(&self, latitude: f64, longitude: f64) -> Option<f32> {
        let x = (latitude - self.xllcorner) / self.cell_size;
        let y = (longitude - self.yllcorner) / self.cell_size;

        let y = (self.height - 1) as f64 - y;

        let fx = x.floor() as usize;
        let fy = y.floor() as usize;
        if x < 0.0 || fx >= self.width - 1 || y < 0.0 || fy >= self.height - 1 {
            return None;
        }

        let h00 = self.values[fx + fy * self.width];
        let h10 = self.values[fx + 1 + fy * self.width];
        let h01 = self.values[fx + (fy + 1) * self.width];
        let h11 = self.values[fx + 1 + (fy + 1) * self.width];
        let h0 = h00 + (h01 - h00) * (y - fy as f64) as f32;
        let h1 = h10 + (h11 - h10) * (y - fy as f64) as f32;
        Some(h0 + (h1 - h0) * (x - fx as f64) as f32)
    }
}

pub(crate) struct RasterCache<F> {
    load: F,
    holes: HashSet<(i16, i16)>,
    rasters: LruCache<(i16, i16), Raster>,
}
impl<F: FnMut(&mut AssetLoadContext, i16, i16) -> Option<Raster>> RasterCache<F> {
    pub fn new(load: F, size: usize) -> Self {
        Self {
            load,
            holes: HashSet::new(),
            rasters: LruCache::new(size),
        }
    }
    pub fn interpolate(
        &mut self,
        context: &mut AssetLoadContext,
        latitude: f64,
        longitude: f64,
    ) -> Option<f32> {
        let key = (latitude.floor() as i16, longitude.floor() as i16);

        if self.holes.contains(&key) {
            return None;
        }
        if let Some(dem) = self.rasters.get_mut(&key) {
            return dem.interpolate(latitude, longitude);
        }

        match (self.load)(context, key.0, key.1) {
            Some(dem) => {
                let elevation = dem.interpolate(latitude, longitude);
                assert!(elevation.is_some());
                self.rasters.insert(key, dem);
                elevation
            }
            None => {
                self.holes.insert(key);
                None
            }
        }
    }
}

/// Currently assumes that values are taken at the *center* of cells.
pub(crate) struct GlobalRaster<T: Into<f64> + Copy> {
    pub width: usize,
    pub height: usize,
    pub bands: usize,
    pub values: Vec<T>,
}
impl<T: Into<f64> + Copy> GlobalRaster<T> {
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
        let h10 = self.get(fx, 1 + fy, band);
        let h01 = self.get(fx, fy + 1, band);
        let h11 = self.get(fx + 1, fy + 1, band);
        let h0 = h00 + (h01 - h00) * (y - fy as f64);
        let h1 = h10 + (h11 - h10) * (y - fy as f64);
        h0 + (h1 - h0) * (x - fx as f64)
    }
}
