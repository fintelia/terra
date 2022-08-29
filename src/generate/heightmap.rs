use anyhow::Error;
use crossbeam::channel::{self, Receiver, Sender};
use futures::future::{BoxFuture, FutureExt};
use lru::LruCache;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Weak};
use types::VFace;

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
                    self.strong.push(n, t.clone());
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
        self.strong.push(n, a);
    }
    fn sender(&self) -> Sender<(K, Arc<T>)> {
        self.sender.clone()
    }
}

// pub(crate) struct HeightmapCache {
//     resolution: usize,
//     border_size: usize,
//     tiles: Cache<VNode, Vec<i16>>,
// }
// impl HeightmapCache {
//     pub fn new(resolution: usize, border_size: usize, capacity: usize) -> Self {
//         Self { resolution, border_size, tiles: Cache::new(capacity) }
//     }

//     pub(crate) fn get_tile<'a>(
//         &mut self,
//         mapfile: &'a MapFile,
//         node: VNode,
//     ) -> BoxFuture<'a, Result<Arc<Vec<i16>>, Error>> {
//         let mut tiles_pending = Vec::new();
//         let mut root = None;

//         let mut n = node;
//         loop {
//             if let Some(t) = self.tiles.get(n) {
//                 root = Some(t);
//                 break;
//             }

//             tiles_pending
//                 .push(async move { (n, mapfile.read_tile(LayerType::Heightmaps, n).await) });
//             match n.parent() {
//                 Some((p, _)) => n = p,
//                 None => break,
//             }
//         }

//         let sender = self.tiles.sender();
//         let (resolution, border_size) = (self.resolution, self.border_size);
//         async move {
//             let tiles = future::join_all(tiles_pending.into_iter()).await;
//             for (n, t) in tiles.into_iter().rev() {
//                 let tile = Arc::new(match root.take() {
//                     None => tilefmt::uncompress_heightmap_tile(None, &*t?.unwrap()).1,
//                     Some(parent_tile) => {
//                         tilefmt::uncompress_heightmap_tile(
//                             Some((
//                                 NODE_OFFSETS[n.parent().unwrap().1 as usize],
//                                 border_size,
//                                 resolution,
//                                 &*parent_tile,
//                             )),
//                             &*t?.unwrap(),
//                         )
//                         .1
//                     }
//                 });
//                 let _ = sender.send((n, Arc::clone(&tile)));
//                 root = Some(tile);
//             }
//             Ok(root.unwrap())
//         }
//         .boxed()
//     }
// }

pub(crate) struct SectorCache<T, F: 'static> {
    sectors: Cache<Sector, Vec<T>>,
    parse: &'static F,
    directory: PathBuf,
    filename_extension: &'static str,
}
impl<T, F> SectorCache<T, F>
where
    F: Fn(&[u8]) -> Result<Vec<T>, Error> + Send + Sync + 'static,
    T: 'static + Send + Sync,
{
    pub fn new(directory: PathBuf, f: &'static F) -> Self {
        Self { sectors: Cache::new(32), directory, filename_extension: "tiff", parse: f }
    }

    pub(crate) fn get_sector<'a>(
        &mut self,
        s: Sector,
        level: Option<u8>,
    ) -> BoxFuture<'a, Result<Arc<Vec<T>>, Error>> {
        if let Some(sector) = self.sectors.get(s) {
            return futures::future::ready(Ok(sector)).boxed();
        }

        let path = match level {
            Some(level) => self.directory.join(&format!(
                "{}_S-{}-{:02}x{:02}.{}",
                VFace(s.face),
                level,
                s.x,
                s.y,
                self.filename_extension
            )),
            None => self.directory.join(&format!(
                "S-{}-{}x{}.{}",
                VFace(s.face),
                s.x,
                s.y,
                self.filename_extension
            )),
        };

        // tilefmt::uncompress_heightmap_tile(None, &bytes).1
        let parse = self.parse.clone();
        let sender = self.sectors.sender();
        async move {
            let bytes = tokio::fs::read(&path).await.expect(&format!("{:?}", path));
            tokio::task::spawn_blocking(move || {
                let sector = Arc::new(parse(&bytes)?);
                let _ = sender.send((s, Arc::clone(&sector)));
                Ok(sector)
            })
            .await?
        }
        .boxed()
    }
}
