use crate::coordinates;
use anyhow::Error;
use bincode::config::WithOtherEndian;
use bit_vec::BitVec;
use crossbeam::channel::{self, Receiver, Sender};
use futures::channel::oneshot;
use futures::future::BoxFuture;
use futures::stream::StreamExt;
use futures::{Future, FutureExt};
use lru::LruCache;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::convert::TryFrom;
use std::f64::consts::PI;
use std::mem;
use std::ops::Index;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, Weak};

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
impl<T> std::fmt::Debug for Raster<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Raster")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("bands", &self.bands)
            .field("cell_size", &self.cell_size)
            .field("latitude_llcorner", &self.latitude_llcorner)
            .field("longitude_llcorner", &self.longitude_llcorner)
            .finish()
    }
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
        let y = (self.latitude_llcorner + self.cell_size * self.height as f64 - latitude)
            / self.cell_size;

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
        let y = (self.latitude_llcorner + self.cell_size * self.height as f64 - latitude)
            / self.cell_size;

        let fx = x.floor() as usize;
        let fy = y.floor() as usize;

        if x < 0.0 || fx >= self.width || y < 0.0 || fy >= self.height {
            return None;
        }

        Some(self.values[(fx + fy * self.width) * self.bands + band].into())
    }
}

type ParseRasterFunction<T> = dyn Fn(i16, i16, &[u8]) -> Result<Raster<T>, Error> + Send + Sync;

// pub(crate) struct RasterCache<T: Send + Sync + 'static> {
//     raster_size_degrees: u16,
//     parse: &'static ParseRasterFunction<T>,
//     filenames: Box<[Option<PathBuf>]>,
//     weak: Box<[Option<Weak<Raster<T>>>]>,
//     strong: LruCache<u16, Arc<Raster<T>>>,
//     sender: Sender<(u16, Arc<Raster<T>>)>,
//     receiver: Receiver<(u16, Arc<Raster<T>>)>,
// }
// impl<T: Send + Sync + 'static> RasterCache<T> {
//     pub fn new(
//         filenames: Box<[Option<PathBuf>]>,
//         raster_size_degrees: u16,
//         capacity: usize,
//         parse: &'static ParseRasterFunction<T>,
//     ) -> Self {
//         let (sender, receiver) = channel::unbounded();

//         Self {
//             parse,
//             raster_size_degrees,
//             weak: vec![None; filenames.len()].into_boxed_slice(),
//             strong: LruCache::new(capacity),
//             filenames,
//             sender,
//             receiver,
//         }
//     }

//     fn insert(&mut self, key: u16, raster: Arc<Raster<T>>) {
//         self.weak[key as usize] = Some(Arc::downgrade(&raster));
//         self.strong.insert(key, raster);
//     }

//     fn try_get(&mut self, key: u16) -> Option<Option<Arc<Raster<T>>>> {
//         if self.filenames[key as usize].is_none() {
//             return Some(None);
//         }

//         let mut found = None;
//         while let Ok(t) = self.receiver.try_recv() {
//             if t.0 == key {
//                 found = Some(t.1.clone());
//             }
//             self.insert(t.0, t.1);
//         }
//         if found.is_some() {
//             return Some(found);
//         }

//         match self.strong.get_mut(&key) {
//             Some(e) => Some(Some(Arc::clone(e))),
//             None => match self.weak[key as usize].as_ref().and_then(|w| w.upgrade()) {
//                 Some(t) => {
//                     self.strong.insert(key, t.clone());
//                     Some(Some(Arc::clone(&t)))
//                 }
//                 None => {
//                     self.weak[key as usize] = None;
//                     None
//                 }
//             },
//         }
//     }

//     pub fn get(
//         &mut self,
//         latitude: i16,
//         longitude: i16,
//     ) -> BoxFuture<'static, Result<Option<Arc<Raster<T>>>, Error>> {
//         let y = u16::try_from(latitude + 90).unwrap() / self.raster_size_degrees;
//         let x = u16::try_from(longitude + 180).unwrap() / self.raster_size_degrees;
//         let key = y * 360 / self.raster_size_degrees as u16 + x;

//         let rs = self.raster_size_degrees as i16;
//         let latitude = latitude - (latitude % rs + rs) % rs;
//         let longitude = longitude - (longitude % rs + rs) % rs;

//         assert!(key < 18 * 36, "{} {} key={}", x, y, key);

//         if let Some(raster) = self.try_get(key) {
//             return futures::future::ready(Ok(raster)).boxed();
//         }

//         match &self.filenames[key as usize] {
//             Some(ref filename) => {
//                 let parse = (&self.parse).clone();
//                 let sender = self.sender.clone();
//                 let file = tokio::fs::read(filename.clone());
//                 let filename = filename.clone();
//                 async move {
//                     let raster = Arc::new(
//                         parse(latitude, longitude, &file.await?)
//                             .map_err(|_| anyhow::format_err!("{:?}", filename))?,
//                     );
//                     sender.send((key, raster.clone()))?;
//                     Ok(Some(raster))
//                 }
//                 .boxed()
//             }
//             None => futures::future::ready(Ok(None)).boxed(),
//         }
//     }
// }

// struct Reservation<T: Send + Sync + 'static> {
//     rasters: HashMap<u16, Arc<Raster<T>>>,
//     needed_rasters: VecDeque<u16>,
//     run: Box<dyn FnMut(&HashMap<u16, Arc<Raster<T>>>)>,
// }

pub(crate) type RasterWorkFunc<T> = Box<
    dyn FnOnce(
            fnv::FnvHashMap<(i16, i16), Option<&Raster<T>>>,
        ) -> BoxFuture<Result<(), anyhow::Error>>
        + Send
        + 'static,
>;

enum RasterCacheEntry<T: Send + Sync + 'static> {
    Empty,
    Pending(Vec<oneshot::Sender<Arc<Raster<T>>>>),
    Loaded(Arc<Raster<T>>),
}

struct RasterCacheInner<T: Send + Sync + 'static> {
    parse: &'static ParseRasterFunction<T>,
    raster_size_degrees: u16,
    lru: LruCache<usize, Arc<Raster<T>>>,
    rasters: Box<[RasterCacheEntry<T>]>,
    reserved_slots: usize,
    filenames: Box<[Option<PathBuf>]>,
    capacity: usize,
}

pub(crate) struct RasterCache<T: Send + Sync + 'static> {
    inner: Arc<Mutex<RasterCacheInner<T>>>,
    pub raster_size_degrees: u16,

    unstarted:
        VecDeque<(Vec<(i16, i16)>, RasterWorkFunc<T>, oneshot::Sender<Result<(), anyhow::Error>>)>,
}
impl<T: Send + Sync + 'static> RasterCacheInner<T> {
    pub(crate) fn key(raster_size: u16, latitude: i16, longitude: i16) -> usize {
        let y = u16::try_from(latitude + 90).unwrap() / raster_size;
        let x = u16::try_from(longitude + 180).unwrap() / raster_size;
        usize::from(y * 360 / raster_size as u16 + x)
    }

    fn count_needed(&self, coordinates: &[(i16, i16)]) -> usize {
        let mut needed_slots = 0;
        for &(latitude, longitude) in coordinates {
            let key = Self::key(self.raster_size_degrees, latitude, longitude);
            if let RasterCacheEntry::Empty = self.rasters[key as usize] {
                if self.filenames[key as usize].is_some() {
                    needed_slots += 1;
                }
            }
        }
        needed_slots
    }

    fn execute_inner(
        &mut self,
        arc: &Arc<Mutex<Self>>,
        coordinates: Vec<(i16, i16)>,
        f: RasterWorkFunc<T>,
    ) -> impl Future<Output = Result<(), anyhow::Error>> {
        let keys: Vec<_> = coordinates
            .iter()
            .map(|&(lat, long)| Self::key(self.raster_size_degrees, lat, long))
            .collect();

        // Promote all needed in rasters in lru into Loaded entries.
        for &key in &keys {
            if let Some(raster) = self.lru.pop(&key) {
                assert!(matches!(self.rasters[key], RasterCacheEntry::Empty));
                self.rasters[key] = RasterCacheEntry::Loaded(raster);
                self.reserved_slots += 1;
            }
        }

        // Gather futures for the loading of rasters.
        let raster_futures: Vec<_> = coordinates.iter().zip(&keys).map(|(&(latitude, longitude), &key)| {
            match &mut self.rasters[key] {
                RasterCacheEntry::Pending(ref mut queue) => {
                    let (sender, receiver) = oneshot::channel();
                    queue.push(sender);
                    async { Ok::<_, anyhow::Error>(Some(receiver.await?)) }.boxed()
                }
                RasterCacheEntry::Loaded(raster) => {
                    futures::future::ready(Ok(Some(raster.clone()))).boxed()
                }
                RasterCacheEntry::Empty => match &self.filenames[key as usize] {
                    Some(ref filename) => {
                        assert!(matches!(self.rasters[key], RasterCacheEntry::Empty));
                        self.rasters[usize::from(key)] = RasterCacheEntry::Pending(Vec::new());
                        self.reserved_slots += 1;
                        while self.reserved_slots + self.lru.len() > self.capacity {
                            self.lru.pop_lru().unwrap();
                        }

                        let parse = (&self.parse).clone();
                        let file = tokio::fs::read(filename.clone());
                        let filename = filename.clone();

                        let rs = self.raster_size_degrees as i16;
                        let latitude = latitude - (latitude % rs + rs) % rs;
                        let longitude = longitude - (longitude % rs + rs) % rs;
                        let arc = arc.clone();
                        async move {
                            // Load and parse raster.
                            let raster = Arc::new(
                                parse(latitude, longitude, &file.await?)
                                    .map_err(|_| anyhow::format_err!("{:?}", filename))?,
                            );
                            // Insert into rasters map.
                            let mut inner = arc.lock().unwrap();
                            match &mut inner.rasters[usize::from(key)] {
                                RasterCacheEntry::Pending(v) => {
                                    for sender in v.drain(..) {
                                        sender.send(raster.clone()).unwrap();
                                    }
                                }
                                _ => unreachable!(),
                            };
                            inner.rasters[usize::from(key)] =
                                RasterCacheEntry::Loaded(raster.clone());
                            Ok(Some(raster))
                        }
                        .boxed()
                    }
                    None => futures::future::ready(Ok(None)).boxed(),
                },
            }
        }).collect();

        let inner = arc.clone();
        async move {
            let rasters: Vec<Option<Arc<Raster<T>>>> = futures::future::join_all(raster_futures)
                .await
                .into_iter()
                .collect::<Result<_, anyhow::Error>>()?;
            let map: fnv::FnvHashMap<_, Option<&Raster<T>>> = coordinates
                .iter()
                .copied()
                .zip(rasters.iter().map(|o: &Option<Arc<_>>| o.as_deref()))
                .collect();

            let ret: Result<(), anyhow::Error> = f(map).await;

            for (raster, key) in rasters.into_iter().zip(keys) {
                // See if cache entry can be freed.
                if let Some(arc) = raster {
                    if Arc::strong_count(&arc) != 2 {
                        continue;
                    }
                    let mut inner = inner.lock().unwrap();
                    if Arc::strong_count(&arc) != 2 {
                        continue;
                    }
                    assert!(matches!(inner.rasters[key], RasterCacheEntry::Loaded(_)));
                    inner.rasters[usize::from(key)] = RasterCacheEntry::Empty;
                    inner.lru.push(key, arc);
                    inner.reserved_slots -= 1;
                }
            }

            ret
        }
    }
}
impl<T: Send + Sync + 'static> RasterCache<T> {
    pub fn new(
        filenames: Box<[Option<PathBuf>]>,
        raster_size_degrees: u16,
        capacity: usize,
        parse: &'static ParseRasterFunction<T>,
    ) -> Self {
        Self {
            inner: Arc::new(Mutex::new(RasterCacheInner {
                parse,
                raster_size_degrees,
                lru: LruCache::unbounded(),
                rasters: (0..filenames.len()).map(|_| RasterCacheEntry::Empty).collect(),
                reserved_slots: 0,
                filenames,
                capacity,
            })),
            raster_size_degrees,
            unstarted: Default::default(),
        }
    }

    pub(crate) fn execute_with(
        &mut self,
        coordinates: Vec<(i16, i16)>,
        f: RasterWorkFunc<T>,
    ) -> impl Future<Output = Result<(), anyhow::Error>> {
        let mut inner = self.inner.lock().unwrap();

        // If no loading is required, simply run the function now.
        {
            let needed = inner.count_needed(&*coordinates);
            assert!(needed <= inner.capacity, "{} {}", needed, inner.capacity);
            if needed == 0
                || self.unstarted.is_empty() && needed + inner.reserved_slots <= inner.capacity
            {
                return inner.execute_inner(&self.inner, coordinates, f).boxed();
            }
        }

        // Loop over the unstarted work items, and see how many are now able to be started.
        let mut unordered = futures::stream::FuturesUnordered::new();
        while !self.unstarted.is_empty() {
            let needed = inner.count_needed(&self.unstarted.front().as_ref().unwrap().0);
            if needed + inner.reserved_slots <= inner.capacity {
                let (coordinates, f, sender) = self.unstarted.pop_front().unwrap();
                unordered.push(
                    inner
                        .execute_inner(&self.inner, coordinates, f)
                        .map(|r| {
                            sender.send(r).unwrap();
                            Ok(())
                        })
                        .boxed(),
                );
            } else {
                break;
            }
        }

        // If we've started all other pending work and there's enough capacity left, then start the requested work.
        if self.unstarted.is_empty()
            && inner.count_needed(&*coordinates) + inner.reserved_slots <= inner.capacity
        {
            unordered.push(inner.execute_inner(&self.inner, coordinates, f).boxed());
        } else {
            let (sender, receiver) = oneshot::channel();
            self.unstarted.push_back((coordinates, f, sender));
            unordered.push(async { receiver.await.unwrap() }.boxed());
        }

        // Return a future that runs all the queued work in parallel.
        async move {
            while let Some(v) = unordered.next().await {
                v?;
            }
            Ok::<(), anyhow::Error>(())
        }
        .boxed()
    }
}

/// Currently assumes that values are taken at the *center* of cells.
#[allow(unused)]
pub(crate) struct GlobalRaster<T: Into<f64> + Copy, C: Index<usize, Output = T> = Vec<T>> {
    pub width: usize,
    pub height: usize,
    pub bands: usize,
    pub values: C,
}
#[allow(unused)]
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
