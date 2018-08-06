use bit_vec::BitVec;
use lru_cache::LruCache;

use cache::AssetLoadContext;
use coordinates;

use std::cell::RefCell;
use std::collections::HashSet;
use std::f64::consts::PI;
use std::ops::Index;
use std::rc::Rc;

pub trait Scalar: Copy + 'static {
    fn from_f64(f64) -> Self;
    fn to_f64(self) -> f64;
}

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
#[derive(Clone, Serialize, Deserialize)]
pub struct Raster<T: Into<f64> + Copy> {
    pub width: usize,
    pub height: usize,
    pub cell_size: f64,

    pub xllcorner: f64,
    pub yllcorner: f64,

    pub values: Vec<T>,
}

impl<T: Into<f64> + Copy> Raster<T> {
    /// Returns the vertical spacing between cells, in meters.
    pub fn vertical_spacing(&self) -> f64 {
        self.cell_size.to_radians() * coordinates::PLANET_RADIUS
    }

    /// Returns the horizontal spacing between cells, in meters.
    pub fn horizontal_spacing(&self, x: usize) -> f64 {
        self.vertical_spacing()
            * (self.yllcorner + self.cell_size * x as f64)
                .to_radians()
                .cos()
    }

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

    pub fn ambient_occlusion(&self) -> Raster<u8> {
        // See: https://nothings.org/gamedev/horizon

        let mut output = Raster {
            width: self.width,
            height: self.height,
            cell_size: self.cell_size,
            xllcorner: self.xllcorner,
            yllcorner: self.yllcorner,
            values: vec![0; self.width * self.height],
        };

        let mut walk =
            |mut x: usize, mut y: usize, dx: isize, dy: isize, steps: usize, step_size: f64| {
                let mut hull = Vec::new();
                for i in 0..(steps as isize) {
                    let h: f64 = self.values[x + y * self.width].into();
                    if hull.is_empty() {
                        hull.push((-1, h));
                    }

                    while hull.len() >= 2 {
                        let (i1, h1) = hull[hull.len() - 1];
                        let (i2, h2) = hull[hull.len() - 2];
                        if ((h1 - h) * (i - i2) as f64) < ((h2 - h) * (i - i1) as f64) {
                            hull.pop();
                        } else {
                            break;
                        }
                    }

                    let (i1, h1) = hull[hull.len() - 1];
                    let slope = (h1 - h) / ((i - i1) as f64 * step_size);
                    let occlusion: f64 = 1.0 - (slope.atan() / (0.5 * PI)).max(0.0);

                    hull.push((i, h));
                    output.values[x + y * self.width] += (occlusion * 63.75) as u8;
                    x = (x as isize + dx) as usize;
                    y = (y as isize + dy) as usize;
                }
            };

        for x in 0..self.width {
            let spacing = self.horizontal_spacing(x);
            walk(x, 0, 0, 1, self.height, spacing);
            walk(x, self.height - 1, 0, -1, self.height, spacing);
        }
        for y in 0..self.height {
            let spacing = self.vertical_spacing();
            walk(0, y, 1, 0, self.width, spacing);
            walk(self.width - 1, y, -1, 0, self.width, spacing);
        }

        output
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

#[allow(unused)]
pub(crate) struct AmbientOcclusionSource<T: Into<f64> + Copy + 'static>(
    pub(crate) Rc<RefCell<RasterCache<T>>>,
);
impl<T: Into<f64> + Copy + 'static> RasterSource for AmbientOcclusionSource<T> {
    type Type = u8;
    fn load(
        &self,
        context: &mut AssetLoadContext,
        latitude: i16,
        longitude: i16,
    ) -> Option<Raster<u8>> {
        self.0
            .borrow_mut()
            .get(context, latitude, longitude)
            .map(|raster| raster.ambient_occlusion())
    }
}

#[derive(Copy, Clone)]
pub(crate) enum Axis {
    X,
    Y,
}
pub(crate) struct BlurredSource {
    cache: Rc<RefCell<RasterCache<u8>>>,
    sigma: f64,
    axis: Axis,
}
impl BlurredSource {
    pub fn new(cache: Rc<RefCell<RasterCache<u8>>>, sigma: f64) -> Self {
        Self {
            cache: Rc::new(RefCell::new(RasterCache::new(
                Box::new(Self {
                    cache,
                    axis: Axis::X,
                    sigma,
                }),
                512,
            ))),
            axis: Axis::Y,
            sigma,
        }
    }
}
impl RasterSource for BlurredSource {
    type Type = u8;
    fn load(
        &self,
        context: &mut AssetLoadContext,
        latitude: i16,
        longitude: i16,
    ) -> Option<Raster<u8>> {
        let mut values = Vec::new();

        let guassian = |x: f64| f64::exp(-0.5 * x * x);

        let mut cache = self.cache.borrow_mut();

        let (inner0, inner, inner2) = match self.axis {
            Axis::X => {
                let inner0 = cache.get(context, latitude - 1, longitude).cloned();
                let inner = cache.get(context, latitude, longitude)?.clone();
                let inner2 = cache.get(context, latitude + 1, longitude).cloned();
                (inner0, inner, inner2)
            }
            Axis::Y => {
                let inner0 = cache.get(context, latitude, (longitude - 1) % 180).cloned();
                let inner = cache.get(context, latitude, longitude)?.clone();
                let inner2 = cache.get(context, latitude, (longitude + 1) % 180).cloned();
                (inner0, inner, inner2)
            }
        };

        let max_three_sigma = self.sigma * 3.0 / inner.vertical_spacing();
        if max_three_sigma < 1.0 {
            return Some(inner);
        }

        for adjacent in [&inner0, &inner2].iter() {
            if let Some(ref adjacent) = *adjacent {
                assert_eq!(adjacent.width, inner.width);
                assert_eq!(adjacent.height, inner.height);
                assert_eq!(adjacent.cell_size, inner.cell_size);
            }
        }

        let iwidth = inner.width as isize;
        let iheight = inner.height as isize;
        let step = match self.axis {
            Axis::X => 1,
            Axis::Y => iwidth,
        };
        for y in 0..iheight {
            for x in 0..iwidth {
                let mut v = 0.0;
                let mut d = 0.0;
                let spacing = match self.axis {
                    Axis::X => inner.vertical_spacing(),
                    Axis::Y => inner.horizontal_spacing(x as usize),
                };
                let three_sigma = (self.sigma * 3.0 / spacing).floor() as isize;

                let min_offset = match (inner0.as_ref(), self.axis) {
                    (Some(_), _) => -three_sigma,
                    (None, Axis::X) => -three_sigma.min(x),
                    (None, Axis::Y) => -three_sigma.min(y),
                };
                let max_offset = match (inner2.as_ref(), self.axis) {
                    (Some(_), _) => three_sigma,
                    (None, Axis::X) => three_sigma.min(iwidth - x - 1),
                    (None, Axis::Y) => three_sigma.min(iheight - y - 1),
                };

                for offset in min_offset..=max_offset {
                    let i = x + y * iwidth + offset * step;

                    let (raster, i) = if i < 0 {
                        (inner0.as_ref().unwrap(), (i + iwidth * iheight) as usize)
                    } else if i >= iwidth * iheight {
                        (inner2.as_ref().unwrap(), (i - iwidth * iheight) as usize)
                    } else {
                        (&inner, i as usize)
                    };

                    let g = guassian(offset as f64 * spacing / self.sigma);
                    v += g * raster.values[i] as f64;
                    d += g;
                }
                v /= d;
                assert!(v < 255.5);
                values.push(v as u8);
            }
        }

        Some(Raster { values, ..inner })
    }
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
    pub fn get(
        &mut self,
        context: &mut AssetLoadContext,
        latitude: i16,
        longitude: i16,
    ) -> Option<&mut Raster<T>> {
        let key = (latitude, longitude);
        if self.holes.contains(&key) {
            return None;
        }
        if self.rasters.contains_key(&key) {
            return self.rasters.get_mut(&key);
        }
        match self.source.load(context, key.0, key.1) {
            Some(raster) => {
                self.rasters.insert(key, raster);
                return self.rasters.get_mut(&key);
            }
            None => {
                self.holes.insert(key);
                None
            }
        }
    }
    pub fn interpolate(
        &mut self,
        context: &mut AssetLoadContext,
        latitude: f64,
        longitude: f64,
    ) -> Option<f64> {
        self.get(context, latitude.floor() as i16, longitude.floor() as i16)
            .and_then(|raster| raster.interpolate(latitude, longitude))
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
