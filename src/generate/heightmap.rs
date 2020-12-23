use crate::cache::AssetLoadContext;
use crate::coordinates::CoordinateSystem;
use crate::mapfile::MapFile;
use crate::terrain::dem::DemSource;
use crate::terrain::quadtree::node::VNode;
use crate::terrain::raster::{GlobalRaster, RasterCache};
use crate::terrain::tile_cache::{LayerParams, LayerType};
use anyhow::Error;
use cgmath::Vector2;
use futures::future::{self, BoxFuture, FutureExt};
use lru_cache::LruCache;
use rayon::prelude::*;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::io::{Cursor, Read, Write};
use std::sync::{Arc, Weak};
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use vec_map::VecMap;

fn compress_heightmap_tile(
    resolution: usize,
    skirt: usize,
    log2_scale_factor: i8,
    heights: &[i16],
    parent: Option<(u8, &[i16])>,
) -> Vec<u8> {
    assert_eq!(resolution % 2, 1);
    let half_resolution = resolution / 2 + 1;
    let mut output = Vec::new();
    let mut grid = Vec::new();

    assert!(log2_scale_factor >= 0 && log2_scale_factor <= 12);
    let scale_factor = 1i16 << log2_scale_factor;
    let inv_scale_factor = 1.0 / scale_factor as f32;

    match parent {
        None => {
            for y in 0..half_resolution {
                for x in 0..half_resolution {
                    output.push(
                        (heights[(x * 2) + (y * 2) * resolution] as f32 * inv_scale_factor).round()
                            as i16,
                    );
                }
            }
            grid = output.clone();
        }
        Some((index, parent_heights)) => {
            let offset: Vector2<usize> = Vector2::new(skirt / 2, skirt / 2)
                + crate::terrain::quadtree::node::OFFSETS[index as usize].cast().unwrap()
                    * ((resolution - 2 * skirt) / 2);
            for y in 0..half_resolution {
                for x in 0..half_resolution {
                    let height = heights[(x * 2) + (y * 2) * resolution];
                    let pheight = parent_heights[(offset.x + x) + (offset.y + y) * resolution];
                    let delta =
                        (height.wrapping_sub(pheight) as f32 * inv_scale_factor).round() as i16;
                    output.push(delta);
                    grid.push(pheight.wrapping_add(delta * scale_factor));
                }
            }
        }
    }

    for y in (0..resolution).step_by(2) {
        for x in (1..resolution).step_by(2) {
            let interpolated = (grid[(x - 1) / 2 + y * half_resolution / 2] as i32
                + grid[(x + 1) / 2 + y * half_resolution / 2] as i32)
                / 2;
            output.push(
                (heights[x + y * resolution].wrapping_sub(interpolated as i16) as f32
                    * inv_scale_factor)
                    .round() as i16,
            );
        }
    }

    for y in (1..resolution).step_by(2) {
        for x in (0..resolution).step_by(2) {
            let interpolated = (grid[x / 2 + (y - 1) * half_resolution / 2] as i32
                + grid[x / 2 + (y + 1) * half_resolution / 2] as i32)
                / 2;
            output.push(
                (heights[x + y * resolution].wrapping_sub(interpolated as i16) as f32
                    * inv_scale_factor)
                    .round() as i16,
            );
        }
    }

    for y in (1..resolution).step_by(2) {
        for x in (1..resolution).step_by(2) {
            let interpolated = (grid[(x - 1) / 2 + (y - 1) * half_resolution / 2] as i32
                + grid[(x + 1) / 2 + (y - 1) * half_resolution / 2] as i32
                + grid[(x - 1) / 2 + (y + 1) * half_resolution / 2] as i32
                + grid[(x + 1) / 2 + (y + 1) * half_resolution / 2] as i32)
                / 4;
            output.push(
                (heights[x + y * resolution].wrapping_sub(interpolated as i16) as f32
                    * inv_scale_factor)
                    .round() as i16,
            );
        }
    }

    // let mut v = vec![0; output.len() * 2+2];
    // v[0] = 1;
    // v[1] = 8;
    // v[2..].copy_from_slice(bytemuck::cast_slice(&output));
    // v
    // let mut e = flate2::write::ZlibEncoder::new(vec![0, 8], flate2::Compression::default());
    // e.write_all(bytemuck::cast_slice(&output)).unwrap();
    // e.finish().unwrap()
    let mut e =
        lz4::EncoderBuilder::new().level(9).build(vec![2, log2_scale_factor as u8]).unwrap();
    e.write_all(bytemuck::cast_slice(&output)).unwrap();
    e.finish().0
}

fn uncompress_heightmap_tile(
    resolution: usize,
    skirt: usize,
    parent: Option<(u8, &[i16])>,
    bytes: &[u8],
) -> Vec<i16> {
    let scale_factor = if bytes[0] == 1 {
        bytes[1] as i16
    } else if bytes[0] == 2 {
        assert!(bytes[1] as i8 >= 0 && bytes[1] <= 12);
        1i16 << bytes[1]
    } else {
        panic!("unknown heightmap tile version.");
    };

    let mut encoded = vec![0i16; resolution * resolution];
    // flate2::read::ZlibDecoder::new(Cursor::new(&bytes[2..]))
    //     .read_exact(bytemuck::cast_slice_mut(&mut encoded))
    //     .unwrap();
    lz4::Decoder::new(Cursor::new(&bytes[2..]))
        .unwrap()
        .read_exact(bytemuck::cast_slice_mut(&mut encoded))
        .unwrap();
    // encoded.copy_from_slice(bytemuck::cast_slice(&bytes[2..]));

    let encoded: VecDeque<i16> = encoded.into();
    let mut encoded = encoded.into_iter();

    assert_eq!(resolution % 2, 1);
    let half_resolution = resolution / 2 + 1;
    let mut heights = vec![0; resolution * resolution];

    let mut q_0 = vec![1234; (resolution / 2 + 1) * (resolution / 2 + 1)];
    let mut q_1 = vec![1234; (resolution / 2 + 1) * (resolution / 2)];
    let mut q_2 = vec![1234; (resolution / 2) * (resolution / 2 + 1)];
    let mut q_3 = vec![1234; (resolution / 2) * (resolution / 2)];

    if let Some((index, parent_heights)) = parent {
        let offset: Vector2<usize> = Vector2::new(skirt / 2, skirt / 2)
            + crate::terrain::quadtree::node::OFFSETS[index as usize].cast().unwrap()
                * ((resolution - 2 * skirt) / 2);
        for (y, row) in q_0.chunks_exact_mut(half_resolution).enumerate() {
            let parent_row =
                &parent_heights[(offset.x + (offset.y + y) * resolution)..][..half_resolution];
            for ((h, r), p) in row.iter_mut().zip(&mut encoded).zip(parent_row) {
                *h = p.wrapping_add(r.wrapping_mul(scale_factor));
            }
        }
    } else {
        for (h, r) in q_0.iter_mut().zip(&mut encoded) {
            *h = r.wrapping_mul(scale_factor);
        }
    }

    for (y, row) in q_1.chunks_exact_mut(resolution / 2).enumerate() {
        for (x, (h, r)) in row.iter_mut().zip(&mut encoded).enumerate() {
            let e = x + y * half_resolution;
            let interpolated = (q_0[e] as i32 + q_0[e + 1] as i32) / 2;
            *h = r.wrapping_mul(scale_factor).wrapping_add(interpolated as i16);
        }
    }

    for (y, row) in q_2.chunks_exact_mut(resolution / 2 + 1).enumerate() {
        for (x, (h, r)) in row.iter_mut().zip(&mut encoded).enumerate() {
            let e = x + y * half_resolution;
            let interpolated = (q_0[e] as i32 + q_0[e + half_resolution] as i32) / 2;
            *h = r.wrapping_mul(scale_factor).wrapping_add(interpolated as i16);
        }
    }

    for (y, row) in q_3.chunks_exact_mut(resolution / 2).enumerate() {
        for (x, (h, r)) in row.iter_mut().zip(&mut encoded).enumerate() {
            let e = x + y * half_resolution;
            let interpolated = (q_0[e] as i32
                + q_0[e + 1] as i32
                + q_0[e + half_resolution] as i32
                + q_0[e + half_resolution + 1] as i32)
                / 4;
            *h = r.wrapping_mul(scale_factor).wrapping_add(interpolated as i16);
        }
    }

    assert!(encoded.next().is_none());

    for (y, row) in heights.chunks_exact_mut(resolution).enumerate() {
        if y % 2 == 0 {
            let q0_row = &q_0[(y / 2) * (resolution / 2 + 1)..][..(resolution / 2 + 1)];
            let q1_row = &q_1[(y / 2) * (resolution / 2)..][..(resolution / 2)];
            for (h, (&q0, &q1)) in row.chunks_exact_mut(2).zip(q0_row.iter().zip(q1_row)) {
                h[0] = q0;
                h[1] = q1;
            }
            row[resolution - 1] = q0_row[(resolution - 1) / 2];
        } else {
            let q2_row = &q_2[(y / 2) * (resolution / 2 + 1)..][..(resolution / 2 + 1)];
            let q3_row = &q_3[(y / 2) * (resolution / 2)..][..(resolution / 2)];
            for (h, (&q2, &q3)) in row.chunks_exact_mut(2).zip(q2_row.iter().zip(q3_row)) {
                h[0] = q2;
                h[1] = q3;
            }
            row[resolution - 1] = q2_row[(resolution - 1) / 2];
        }
    }

    assert_eq!(heights[0], q_0[0]);
    assert_eq!(heights[1], q_1[0]);
    assert_eq!(heights[2], q_0[1]);
    assert_eq!(heights[resolution], q_2[0]);
    assert_eq!(heights[resolution + 1], q_3[0]);
    assert_eq!(heights[resolution + 2], q_2[1]);

    heights
}

struct Cache<T: Clone> {
    weak: HashMap<VNode, Weak<T>>,
    strong: VecMap<LruCache<VNode, Arc<T>>>,
    sender: UnboundedSender<(VNode, Arc<T>)>,
    receiver: UnboundedReceiver<(VNode, Arc<T>)>,
    capacity: usize,
}
impl<T: Clone> Cache<T> {
    fn new(capacity: usize) -> Self {
        let (sender, receiver) = mpsc::unbounded_channel();
        Self { weak: HashMap::default(), strong: VecMap::default(), sender, receiver, capacity }
    }
    fn get(&mut self, n: VNode) -> Option<Arc<T>> {
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

        match self.strong.get_mut(n.level() as usize).and_then(|l| l.get_mut(&n)) {
            Some(e) => Some(Arc::clone(&e)),
            None => match self.weak.get(&n)?.upgrade() {
                Some(t) => {
                    let capacity = self.capacity;
                    self.strong
                        .entry(n.level() as usize)
                        .or_insert_with(|| LruCache::new(capacity))
                        .insert(n, t.clone());
                    Some(Arc::clone(&t))
                }
                None => {
                    self.weak.remove(&n);
                    None
                }
            },
        }
    }
    fn insert(&mut self, n: VNode, a: Arc<T>) {
        let capacity = self.capacity;
        self.weak.insert(n, Arc::downgrade(&a));
        self.strong
            .entry(n.level() as usize)
            .or_insert_with(|| LruCache::new(capacity))
            .insert(n, a);
    }
    fn sender(&self) -> UnboundedSender<(VNode, Arc<T>)> {
        self.sender.clone()
    }
}

pub(crate) struct HeightmapCache {
    layer: LayerParams,
    tiles: Cache<Vec<i16>>,
}
impl HeightmapCache {
    pub fn new(layer: LayerParams, capacity: usize) -> Self {
        Self { layer, tiles: Cache::new(capacity) }
    }

    pub(crate) fn get_tile<'a>(
        &mut self,
        mapfile: &'a MapFile,
        node: VNode,
    ) -> BoxFuture<'a, Option<Arc<Vec<i16>>>> {
        let mut tiles_pending = Vec::new();
        let mut root = None;

        let mut n = node;
        loop {
            if let Some(t) = self.tiles.get(n) {
                root = Some(t);
                break;
            }

            tiles_pending.push(async move {
                (n, mapfile.read_tile_async_raw(LayerType::Heightmaps, n).await)
            });
            match n.parent() {
                Some((p, _)) => n = p,
                None => break,
            }
        }

        let sender = self.tiles.sender();
        let layer = self.layer.clone();
        async move {
            let tiles = future::join_all(tiles_pending.into_iter()).await;
            for (n, t) in tiles.into_iter().rev() {
                let tile = Arc::new(match root.take() {
                    None => uncompress_heightmap_tile(
                        layer.texture_resolution as usize,
                        layer.texture_border_size as usize,
                        None,
                        &*t?,
                    ),
                    Some(parent_tile) => uncompress_heightmap_tile(
                        layer.texture_resolution as usize,
                        layer.texture_border_size as usize,
                        Some((n.parent().unwrap().1, &*parent_tile)),
                        &*t?,
                    ),
                });
                let _ = sender.send((n, Arc::clone(&tile)));
                root = Some(tile);
            }
            root
        }
        .boxed()
    }
}

pub(crate) struct HeightmapGen {
    pub tile_cache: HeightmapCache,
    pub dems: RasterCache<f32, Vec<f32>>,
    pub global_dem: GlobalRaster<i16>,
}
impl HeightmapGen {
    pub(crate) async fn generate_heightmaps<'a>(
        &mut self,
        context: &mut AssetLoadContext<'a>,
        mapfile: &MapFile,
        node: VNode,
    ) -> Result<tokio::sync::oneshot::Receiver<Vec<u8>>, Error> {
        let mut parent: Option<(u8, Arc<Vec<i16>>)> = None;
        if let Some((p, i)) = node.parent() {
            parent = Some((i, self.tile_cache.get_tile(mapfile, p).await.unwrap()));
        }

        let layer = &self.tile_cache.layer;
        let coordinates: Vec<_> = (0..(layer.texture_resolution * layer.texture_resolution))
            .into_par_iter()
            .map(|i| {
                let cspace = node.grid_position_cspace(
                    (i % layer.texture_resolution) as i32,
                    (i / layer.texture_resolution) as i32,
                    layer.texture_border_size as u16,
                    layer.texture_resolution as u16,
                );
                let sspace = CoordinateSystem::cspace_to_sspace(cspace);
                let polar = CoordinateSystem::sspace_to_polar(sspace);
                (polar.x.to_degrees(), polar.y.to_degrees())
            })
            .collect();

        let mut heightmap = vec![
            0i16;
            self.tile_cache.layer.texture_resolution as usize
                * self.tile_cache.layer.texture_resolution as usize
        ];

        if node.level() <= 3 {
            let global_dem = &self.global_dem;
            heightmap.par_iter_mut().zip(coordinates.into_par_iter()).for_each(
                |(h, (lat, long))| {
                    *h = global_dem.interpolate(lat, long, 0) as i16;
                },
            );
        } else {
            let mut samples: HashMap<(i16, i16), Vec<(usize, f64, f64)>> = Default::default();
            for (i, (lat, long)) in coordinates.into_iter().enumerate() {
                samples
                    .entry((lat.floor() as i16, long.floor() as i16))
                    .or_default()
                    .push((i, lat, long));
            }

            for (tile, v) in samples {
                if DemSource::Srtm90m.tile_exists(tile.0, tile.1) {
                    let tile = self.dems.get(context, tile.0, tile.1).unwrap();
                    v.into_par_iter()
                        .map(|(i, lat, long)| (i, tile.interpolate(lat, long, 0).unwrap() as i16))
                        .collect::<Vec<_>>()
                        .into_iter()
                        .for_each(|(i, h)| heightmap[i] = h);
                } else {
                    let global_dem = &self.global_dem;
                    v.into_par_iter()
                        .map(|(i, lat, long)| (i, global_dem.interpolate(lat, long, 0) as i16))
                        .collect::<Vec<_>>()
                        .into_iter()
                        .for_each(|(i, h)| heightmap[i] = h);
                }
            }
        }

        let (resolution, border_size) = (
            self.tile_cache.layer.texture_resolution as usize,
            self.tile_cache.layer.texture_border_size as usize,
        );

        let (tx, rx) = tokio::sync::oneshot::channel();
        rayon::spawn(move || {
            tx.send(compress_heightmap_tile(
                resolution,
                border_size,
                2 + crate::generate::TILE_CELL_76M.saturating_sub(node.level()) as i8,
                &*heightmap,
                parent.as_ref().map(|&(i, ref a)| (i, &***a)),
            ))
            .unwrap();
        });

        Ok(rx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::distributions::{Distribution, Uniform};
    use test::Bencher;

    #[test]
    fn compress_roundtrip() {
        let skirt = 8;
        let resolution = 513 + skirt * 2;

        let mut rng = rand::thread_rng();
        let dist = Uniform::from(-1000..8000);

        let parent: Vec<i16> =
            (0..(resolution * resolution)).map(|_| dist.sample(&mut rng)).collect();
        let child: Vec<i16> =
            (0..(resolution * resolution)).map(|_| dist.sample(&mut rng)).collect();

        let bytes = compress_heightmap_tile(resolution, skirt, 3, &*child, Some((0, &*parent)));
        let roundtrip = uncompress_heightmap_tile(resolution, skirt, Some((0, &*parent)), &*bytes);

        for i in 0..(resolution * resolution) {
            assert!(
                (roundtrip[i] - child[i]).abs() <= 4,
                "i={}: roundtrip={} child={}",
                i,
                roundtrip[i],
                child[i]
            );
        }
    }

    #[bench]
    fn bench_compress(b: &mut Bencher) {
        let skirt = 8;
        let resolution = 513 + skirt * 2;

        let mut rng = rand::thread_rng();
        let dist = Uniform::from(-1000..8000);

        let parent: Vec<i16> =
            (0..(resolution * resolution)).map(|_| dist.sample(&mut rng)).collect();
        let child: Vec<i16> =
            (0..(resolution * resolution)).map(|_| dist.sample(&mut rng)).collect();

        b.iter(|| compress_heightmap_tile(resolution, skirt, 3, &*child, Some((0, &*parent))));
    }

    #[bench]
    fn bench_uncompress(b: &mut Bencher) {
        let skirt = 8;
        let resolution = 513 + skirt * 2;

        let mut rng = rand::thread_rng();
        let dist = Uniform::from(-1000..8000);

        let parent: Vec<i16> =
            (0..(resolution * resolution)).map(|_| dist.sample(&mut rng)).collect();
        let child: Vec<i16> =
            (0..(resolution * resolution)).map(|_| dist.sample(&mut rng)).collect();

        let bytes = compress_heightmap_tile(resolution, skirt, 3, &*child, Some((0, &*parent)));
        b.iter(|| uncompress_heightmap_tile(resolution, skirt, Some((0, &*parent)), &*bytes));
    }
}
