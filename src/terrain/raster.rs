use crate::cache::{AssetLoadContext, MMappedAsset};
use crate::coordinates;
use anyhow::Error;
use bit_vec::BitVec;
use lru_cache::LruCache;
use memmap::Mmap;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::f64::consts::PI;
use std::ops::{Deref, Index};

pub trait Scalar: Copy + 'static {
    fn from_f64(_: f64) -> Self;
    fn to_f64(self) -> f64;
}

/// Wrapper around BitVec that converts values to f64's so that it can be uses as a backing for a
/// GlobalRaster.
pub struct BitContainer(pub BitVec<u32>);
impl Index<usize> for BitContainer {
    type Output = u8;
    fn index(&self, i: usize) -> &u8 {
        if self.0[i] {
            &255
        } else {
            &0
        }
    }
}

#[derive(Serialize, Deserialize)]
pub(crate) struct MMappedRasterHeader {
    pub width: usize,
    pub height: usize,
    pub bands: usize,
    pub cell_size: f64,

    pub latitude_llcorner: f64,
    pub longitude_llcorner: f64,
}

/// Currently assumes that values are taken at the lower left corner of each cell.
#[derive(Clone, Serialize, Deserialize)]
pub struct Raster<T: Into<f64> + Copy, C: Deref<Target = [T]> = Vec<T>> {
    pub width: usize,
    pub height: usize,
    pub bands: usize,
    pub cell_size: f64,

    pub latitude_llcorner: f64,
    pub longitude_llcorner: f64,

    pub values: C,
}

impl Raster<u8, Mmap> {
    #[allow(unused)]
    pub(crate) fn from_mmapped_raster<MR: MMappedAsset<Header = MMappedRasterHeader>>(
        asset: MR,
        context: &mut AssetLoadContext,
    ) -> Result<Self, Error> {
        let (header, mmap) = asset.load(context)?;

        Ok(Self {
            width: header.width,
            height: header.height,
            bands: header.bands,
            cell_size: header.cell_size,
            latitude_llcorner: header.latitude_llcorner,
            longitude_llcorner: header.longitude_llcorner,
            values: mmap.make_read_only()?,
        })
    }
}

impl<T: Into<f64> + Copy, C: Deref<Target = [T]>> Raster<T, C> {
    /// Returns the vertical spacing between cells, in meters.
    #[allow(unused)]
    pub fn vertical_spacing(&self) -> f64 {
        self.cell_size.to_radians() * coordinates::PLANET_RADIUS
    }

    /// Returns the horizontal spacing between cells, in meters.
    #[allow(unused)]
    pub fn horizontal_spacing(&self, y: usize) -> f64 {
        self.vertical_spacing()
            * (self.latitude_llcorner + self.cell_size * y as f64).to_radians().cos()
    }

    pub fn interpolate(&self, latitude: f64, longitude: f64, band: usize) -> Option<f64> {
        assert!(band < self.bands);

        let x = (longitude - self.longitude_llcorner) / self.cell_size;
        let y = (self.height - 1) as f64 - (latitude - self.latitude_llcorner) / self.cell_size;

        let fx = x.floor() as usize;
        let fy = y.floor() as usize;

        if x < 0.0 || fx >= self.width || y < 0.0 || fy >= self.height {
            return None;
        }

        // TODO: These should be interpolating across tiles...
        let fx_1 = (fx + 1).min(self.width - 1);
        let fy_1 = (fy + 1).min(self.height - 1);

        let h00 = self.values[(fx + fy * self.width) * self.bands + band].into();
        let h10 = self.values[(fx_1 + fy * self.width) * self.bands + band].into();
        let h01 = self.values[(fx + fy_1 * self.width) * self.bands + band].into();
        let h11 = self.values[(fx_1 + fy_1 * self.width) * self.bands + band].into();
        let h0 = h00 + (h01 - h00) * (y - fy as f64);
        let h1 = h10 + (h11 - h10) * (y - fy as f64);
        Some(h0 + (h1 - h0) * (x - fx as f64))
    }

    pub fn nearest3(&self, latitude: f64, longitude: f64) -> Option<[f64; 3]> {
        assert!(self.bands >= 3);

        let x = (longitude - self.longitude_llcorner) / self.cell_size;
        let y = self.height as f64 - (latitude - self.latitude_llcorner) / self.cell_size;

        let fx = x.floor() as usize;
        let fy = y.floor() as usize;

        if x < 0.0 || fx >= self.width || y < 0.0 || fy >= self.height {
            return None;
        }

        let slice = &self.values[(fx + fy * self.width) * self.bands..][..3];
        Some([slice[0].into(), slice[1].into(), slice[2].into()])
    }
}

pub(crate) trait RasterSource {
    type Type: Into<f64> + Copy;
    type Container: Deref<Target = [Self::Type]>;
    fn load(
        &self,
        latitude: i16,
        longitude: i16,
    ) -> Result<Option<Raster<Self::Type, Self::Container>>, Error>;
    fn bands(&self) -> usize;

    /// Degrees of latitude and longitude covered by each raster.
    fn raster_size(&self) -> i16 {
        1
    }
}

pub(crate) struct RasterCache<T: Into<f64> + Copy, C: Deref<Target = [T]>> {
    source: Box<dyn RasterSource<Type = T, Container = C>>,
    holes: HashSet<(i16, i16)>,
    rasters: LruCache<(i16, i16), Raster<T, C>>,
}
impl<T: Into<f64> + Copy, C: Deref<Target = [T]>> RasterCache<T, C> {
    pub fn new(source: Box<dyn RasterSource<Type = T, Container = C>>, size: usize) -> Self {
        Self { source, holes: HashSet::new(), rasters: LruCache::new(size) }
    }
    pub fn get(
        &mut self,
        latitude: i16,
        longitude: i16,
    ) -> Result<Option<&mut Raster<T, C>>, Error> {
        let rs = self.source.raster_size();
        let key = (latitude - (latitude % rs + rs) % rs, longitude - (longitude % rs + rs) % rs);
        if self.holes.contains(&key) {
            return Ok(None);
        }
        if self.rasters.contains_key(&key) {
            return Ok(self.rasters.get_mut(&key));
        }
        match self.source.load(key.0, key.1)? {
            Some(raster) => {
                self.rasters.insert(key, raster);
                return Ok(self.rasters.get_mut(&key));
            }
            None => {
                self.holes.insert(key);
                Ok(None)
            }
        }
    }
    #[allow(unused)]
    pub fn interpolate(
        &mut self,
        latitude: f64,
        longitude: f64,
        band: usize,
    ) -> Result<Option<f64>, Error> {
        Ok(self
            .get(latitude.floor() as i16, longitude.floor() as i16)?
            .and_then(|raster| raster.interpolate(latitude, longitude, band)))
    }
    #[allow(unused)]
    pub fn nearest3(&mut self, latitude: f64, longitude: f64) -> Result<Option<[f64; 3]>, Error> {
        Ok(self
            .get(latitude.floor() as i16, longitude.floor() as i16)?
            .and_then(|raster| raster.nearest3(latitude, longitude)))
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
    #[allow(unused)]
    pub fn spacing(&self) -> f64 {
        let sx = 2.0 * PI * coordinates::PLANET_RADIUS / self.width as f64;
        let sy = PI * coordinates::PLANET_RADIUS / self.height as f64;
        sx.min(sy)
    }

    fn get(&self, x: i64, y: i64, band: usize) -> f64 {
        let y = y.max(0).min(self.height as i64 - 1) as usize;
        let x = (((x % self.width as i64) + self.width as i64) % self.width as i64) as usize;
        self.values[(x + y * self.width) * self.bands + band].into()
    }

    pub fn interpolate(&self, latitude: f64, longitude: f64, band: usize) -> f64 {
        assert!(latitude >= -90.0 && latitude <= 90.0);
        assert!(longitude >= -180.0 && latitude <= 180.0);

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
