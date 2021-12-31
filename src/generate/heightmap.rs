use crate::cache::LayerType;
use crate::coordinates;
use crate::mapfile::MapFile;
use crate::terrain::quadtree::node::VNode;
use crate::terrain::raster::{GlobalRaster, Raster, RasterCache};
use crate::types::VFace;
use anyhow::Error;
use atomicwrites::{AtomicFile, OverwriteBehavior};
use crossbeam::channel::{self, Receiver, Sender};
use futures::future::{self, BoxFuture, FutureExt};
use lru_cache::LruCache;
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::sync::{Arc, Weak};

#[derive(Copy, Clone, Hash, Eq, PartialEq, Debug)]
pub(crate) struct Sector {
    pub face: u8,
    pub x: u32,
    pub y: u32,
}

struct Cache<K: Eq + std::hash::Hash + Copy, T> {
    weak: HashMap<K, Weak<T>>,
    strong: LruCache<K, Arc<T>>,
    sender: Sender<(K, Arc<T>)>,
    receiver: Receiver<(K, Arc<T>)>,
}
impl<K: std::hash::Hash + Eq + Copy, T> Cache<K, T> {
    fn new(capacity: usize) -> Self {
        let (sender, receiver) = channel::unbounded();
        Self { weak: HashMap::default(), strong: LruCache::new(capacity), sender, receiver }
    }
    fn get(&mut self, n: K) -> Option<Arc<T>> {
        let mut found = None;
        while let Ok(t) = self.receiver.try_recv() {
            if t.0 == n {
                found = Some(Arc::clone(&t.1));
            }
            self.insert(t.0, t.1);
        }
        if found.is_some() {
            return found;
        }

        match self.strong.get_mut(&n) {
            Some(e) => Some(Arc::clone(&e)),
            None => match self.weak.get(&n)?.upgrade() {
                Some(t) => {
                    self.strong.insert(n, t.clone());
                    Some(Arc::clone(&t))
                }
                None => {
                    self.weak.remove(&n);
                    None
                }
            },
        }
    }
    fn insert(&mut self, n: K, a: Arc<T>) {
        self.weak.insert(n, Arc::downgrade(&a));
        self.strong.insert(n, a);
    }
    fn sender(&self) -> Sender<(K, Arc<T>)> {
        self.sender.clone()
    }
}

pub(crate) struct HeightmapCache {
    resolution: usize,
    border_size: usize,
    tiles: Cache<VNode, Vec<i16>>,
}
impl HeightmapCache {
    pub fn new(resolution: usize, border_size: usize, capacity: usize) -> Self {
        Self { resolution, border_size, tiles: Cache::new(capacity) }
    }

    pub(crate) fn get_tile<'a>(
        &mut self,
        mapfile: &'a MapFile,
        node: VNode,
    ) -> BoxFuture<'a, Result<Arc<Vec<i16>>, Error>> {
        let mut tiles_pending = Vec::new();
        let mut root = None;

        let mut n = node;
        loop {
            if let Some(t) = self.tiles.get(n) {
                root = Some(t);
                break;
            }

            tiles_pending
                .push(async move { (n, mapfile.read_tile(LayerType::Heightmaps, n).await) });
            match n.parent() {
                Some((p, _)) => n = p,
                None => break,
            }
        }

        let sender = self.tiles.sender();
        let (resolution, border_size) = (self.resolution, self.border_size);
        async move {
            let tiles = future::join_all(tiles_pending.into_iter()).await;
            for (n, t) in tiles.into_iter().rev() {
                let tile = Arc::new(match root.take() {
                    None => tilefmt::uncompress_heightmap_tile(None, &*t?).1,
                    Some(parent_tile) => {
                        tilefmt::uncompress_heightmap_tile(
                            Some((
                                crate::terrain::quadtree::node::OFFSETS
                                    [n.parent().unwrap().1 as usize],
                                border_size,
                                resolution,
                                &*parent_tile,
                            )),
                            &*t?,
                        )
                        .1
                    }
                });
                let _ = sender.send((n, Arc::clone(&tile)));
                root = Some(tile);
            }
            Ok(root.unwrap())
        }
        .boxed()
    }
}

pub(crate) struct SectorCache {
    sectors: Cache<Sector, Vec<i16>>,
}
impl SectorCache {
    pub fn new(capacity: usize) -> Self {
        Self { sectors: Cache::new(capacity) }
    }

    pub(crate) fn get_sector<'a>(
        &mut self,
        nasadem_reprojected_directory: &Path,
        s: Sector,
    ) -> BoxFuture<'a, Result<Arc<Vec<i16>>, Error>> {
        if let Some(sector) = self.sectors.get(s) {
            return futures::future::ready(Ok(sector)).boxed();
        }

        let path = nasadem_reprojected_directory.join(&format!(
            "nasadem_S-{}-{}x{}.raw",
            VFace(s.face),
            s.x,
            s.y
        ));
        let sender = self.sectors.sender();
        async move {
            let bytes = tokio::fs::read(path).await?;
            tokio::task::spawn_blocking(move || {
                let sector = Arc::new(tilefmt::uncompress_heightmap_tile(None, &bytes).1);
                let _ = sender.send((s, Arc::clone(&sector)));
                Ok(sector)
            })
            .await?
        }
        .boxed()
    }
}

pub(crate) struct HeightmapSectorGen {
    pub root_resolution: usize,
    pub root_border_size: usize,
    pub sector_resolution: usize,
    pub dems: RasterCache<f32, Vec<f32>>,
    pub global_dem: Arc<GlobalRaster<i16>>,
}
impl HeightmapSectorGen {
    pub(crate) fn generate_sector(
        &mut self,
        root_node: VNode,
        x: usize,
        y: usize,
        output_file: PathBuf,
    ) -> (usize, BoxFuture<'static, Result<usize, Error>>) {
        // Reproject coordinates
        let coordinates: Vec<_> = (0..(self.sector_resolution * self.sector_resolution))
            .into_par_iter()
            .map(|i| {
                let cspace = root_node.grid_position_cspace(
                    (x * (self.sector_resolution - 1) + (i % self.sector_resolution)) as i32,
                    (y * (self.sector_resolution - 1) + (i / self.sector_resolution)) as i32,
                    self.root_border_size as u32,
                    self.root_resolution as u32,
                );
                let polar = coordinates::cspace_to_polar(cspace);
                (polar.x.to_degrees(), polar.y.to_degrees())
            })
            .collect();

        // Asynchronously start loading required tiles
        let mut tiles: Vec<_> = coordinates
            .par_iter()
            .map(|(lat, long)| (lat.floor() as i16, long.floor() as i16))
            .collect();

        tiles.dedup();
        tiles.sort();
        tiles.dedup();

        let mut rasters = Vec::new();
        for tile in tiles {
            rasters.push(
                self.dems.get(tile.0, tile.1).map(move |f| -> Result<_, Error> { Ok((tile, f?)) }),
            );
        }

        let num_rasters = rasters.len();
        let global_dem = self.global_dem.clone();
        let resolution = self.sector_resolution;
        let fut = async move {
            let mut heightmap = vec![0i16; resolution * resolution];

            let rasters: fnv::FnvHashMap<(i16, i16), Arc<Raster<_, _>>> =
                futures::future::try_join_all(rasters)
                    .await?
                    .into_iter()
                    .filter_map(|v: ((i16, i16), Option<Arc<Raster<_, _>>>)| Some((v.0, v.1?)))
                    .collect();

            heightmap.par_iter_mut().zip(coordinates.into_par_iter()).for_each(
                |(h, (lat, long))| {
                    if let Some(r) = rasters.get(&(lat.floor() as i16, long.floor() as i16)) {
                        *h = r.interpolate(lat, long, 0).unwrap() as i16;
                    }
                    if *h == 0 {
                        *h = global_dem.interpolate(lat, long, 0) as i16;
                    }
                },
            );

            tokio::task::spawn_blocking(move || {
                let tile = tilefmt::compress_heightmap_tile(
                    resolution,
                    0, // 2 + VNode::LEVEL_CELL_76M.saturating_sub(node.level()) as i8,
                    &*heightmap,
                    None, //parent.as_ref().map(|&(i, ref a)| (i, &***a)),
                    7,
                );

                AtomicFile::new(output_file, OverwriteBehavior::AllowOverwrite)
                    .write(|f| f.write_all(&*tile))
            })
            .await??;
            Ok(num_rasters)
        }
        .boxed();

        (num_rasters, fut)
    }
}
