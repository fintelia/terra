use crate::coordinates;
use anyhow::Error;
use bit_vec::BitVec;
use crossbeam::channel::{self, Receiver, Sender};
use futures::future::BoxFuture;
use futures::FutureExt;
use lru_cache::LruCache;
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;
use std::f64::consts::PI;
use std::ops::Index;
use std::path::PathBuf;
use std::sync::{Arc, Weak};

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
pub struct Raster<T> {
    pub width: usize,
    pub height: usize,
    pub bands: usize,
    pub cell_size: f64,

    pub latitude_llcorner: f64,
    pub longitude_llcorner: f64,

    pub values: Vec<T>,
}

impl<T: Into<f64> + Copy> Raster<T> {
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
        let y = (self.latitude_llcorner + self.cell_size * self.height as f64 - latitude) / self.cell_size;

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

    /// Assume cell registration
    pub fn nearest(&self, latitude: f64, longitude: f64, band: usize) -> Option<f64> {
        assert!(self.bands >= band);

        let x = (longitude - self.longitude_llcorner) / self.cell_size;
        let y = (self.latitude_llcorner + self.cell_size * self.height as f64 - latitude) / self.cell_size;

        let fx = x.floor() as usize;
        let fy = y.floor() as usize;

        if x < 0.0 || fx >= self.width || y < 0.0 || fy >= self.height {
            return None;
        }

        Some(self.values[(fx + fy * self.width) * self.bands + band].into())
    }
}

type ParseRasterFunction<T> = dyn Fn(i16, i16, &[u8]) -> Result<Raster<T>, Error> + Send + Sync;

pub(crate) struct RasterCache<T: Send + Sync + 'static> {
    raster_size_degrees: u16,
    parse: &'static ParseRasterFunction<T>,
    filenames: Box<[Option<PathBuf>]>,
    weak: Box<[Option<Weak<Raster<T>>>]>,
    strong: LruCache<u16, Arc<Raster<T>>>,
    sender: Sender<(u16, Arc<Raster<T>>)>,
    receiver: Receiver<(u16, Arc<Raster<T>>)>,
}
impl<T: Send + Sync + 'static> RasterCache<T> {
    pub fn new(
        filenames: Box<[Option<PathBuf>]>,
        raster_size_degrees: u16,
        capacity: usize,
        parse: &'static ParseRasterFunction<T>,
    ) -> Self {
        let (sender, receiver) = channel::unbounded();

        Self {
            parse,
            raster_size_degrees,
            weak: vec![None; filenames.len()].into_boxed_slice(),
            strong: LruCache::new(capacity),
            filenames,
            sender,
            receiver,
        }
    }

    fn insert(&mut self, key: u16, raster: Arc<Raster<T>>) {
        self.weak[key as usize] = Some(Arc::downgrade(&raster));
        self.strong.insert(key, raster);
    }

    fn try_get(&mut self, key: u16) -> Option<Option<Arc<Raster<T>>>> {
        if self.filenames[key as usize].is_none() {
            return Some(None);
        }

        let mut found = None;
        while let Ok(t) = self.receiver.try_recv() {
            if t.0 == key {
                found = Some(t.1.clone());
            }
            self.insert(t.0, t.1);
        }
        if found.is_some() {
            return Some(found);
        }

        match self.strong.get_mut(&key) {
            Some(e) => Some(Some(Arc::clone(e))),
            None => match self.weak[key as usize].as_ref().and_then(|w| w.upgrade()) {
                Some(t) => {
                    self.strong.insert(key, t.clone());
                    Some(Some(Arc::clone(&t)))
                }
                None => {
                    self.weak[key as usize] = None;
                    None
                }
            },
        }
    }

    pub fn get(
        &mut self,
        latitude: i16,
        longitude: i16,
    ) -> BoxFuture<'static, Result<Option<Arc<Raster<T>>>, Error>> {
        let y = u16::try_from(latitude + 90).unwrap() / self.raster_size_degrees;
        let x = u16::try_from(longitude + 180).unwrap() / self.raster_size_degrees;
        let key = y * 360 / self.raster_size_degrees as u16 + x;

        let rs = self.raster_size_degrees as i16;
        let latitude = latitude - (latitude % rs + rs) % rs;
        let longitude = longitude - (longitude % rs + rs) % rs;

        assert!(key < 18 * 36, "{} {} key={}", x, y, key);

        if let Some(raster) = self.try_get(key) {
            return futures::future::ready(Ok(raster)).boxed();
        }

        match &self.filenames[key as usize] {
            Some(ref filename) => {
                let parse = (&self.parse).clone();
                let sender = self.sender.clone();
                let file = tokio::fs::read(filename.clone());
                let filename = filename.clone();
                async move {
                    let raster = Arc::new(parse(latitude, longitude, &file.await?).map_err(|_| anyhow::format_err!("{:?}", filename))?);
                    sender.send((key, raster.clone()))?;
                    Ok(Some(raster))
                }
                .boxed()
            }
            None => futures::future::ready(Ok(None)).boxed(),
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
