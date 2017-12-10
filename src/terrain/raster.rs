use lru_cache::LruCache;

use std::collections::HashSet;

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
    pub fn crop(&self, width: usize, height: usize) -> Self {
        assert!(width > 0 && width <= self.width);
        assert!(height > 0 && height <= self.height);

        let xoffset = (self.width - width) / 2;
        let yoffset = (self.height - height) / 2;

        let mut values = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                values.push(self.values[(x + xoffset) + (y + yoffset) * self.width]);
            }
        }

        Self {
            width,
            height,
            cell_size: self.cell_size,
            xllcorner: self.xllcorner + self.cell_size * (xoffset as f64),
            yllcorner: self.yllcorner + self.cell_size * (yoffset as f64),
            values,
        }
    }

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

pub struct RasterCache<F> {
    load: F,
    holes: HashSet<(i16, i16)>,
    rasters: LruCache<(i16, i16), Raster>,
}
impl<F: FnMut(i16, i16) -> Option<Raster>> RasterCache<F> {
    pub fn new(load: F, size: usize) -> Self {
        Self {
            load,
            holes: HashSet::new(),
            rasters: LruCache::new(size),
        }
    }
    pub fn interpolate(&mut self, latitude: f64, longitude: f64) -> Option<f32> {
        let key = (latitude.floor() as i16, longitude.floor() as i16);

        if self.holes.contains(&key) {
            return None;
        }
        if let Some(dem) = self.rasters.get_mut(&key) {
            return dem.interpolate(latitude, longitude);
        }

        match (self.load)(key.0, key.1) {
            Some(dem) => {
                let elevation = dem.interpolate(latitude, longitude);
                // TODO: assert elevation is some...
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
