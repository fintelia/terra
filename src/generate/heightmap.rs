use anyhow::Error;
use lru::LruCache;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Weak, Mutex, Condvar};

#[derive(Copy, Clone, Hash, Eq, PartialEq, Debug)]
pub(crate) struct Sector {
    pub face: u8,
    pub x: u32,
    pub y: u32,
}

struct Cache<K: Eq + std::hash::Hash + Copy, T> {
    weak: HashMap<K, Weak<T>>,
    strong: LruCache<K, Arc<T>>,
    pending: HashSet<K>,
}
impl<K: std::hash::Hash + Eq + Copy, T> Cache<K, T> {
    fn new(capacity: usize) -> Self {
        Self { weak: HashMap::default(), strong: LruCache::new(capacity), pending: HashSet::default() }
    }
    fn get(&mut self, n: K) -> Option<Arc<T>> {
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
    fn is_pending(&self, n: &K) -> bool {
        self.pending.contains(n)
    }
    fn mark_pending(&mut self, n: K) {
        self.pending.insert(n);
    }
    fn insert(&mut self, n: K, a: Arc<T>) {
        self.weak.insert(n, Arc::downgrade(&a));
        self.strong.push(n, a);
        self.pending.remove(&n);
    }
}

pub(crate) struct CogTileCache {
    cache: Mutex<Cache<(u8, u8, u8, u32), Option<Vec<u8>>>>,
    condvars: Vec<Condvar>,
    cogs: Vec<Vec<cogbuilder::CogBuilder>>,
}
impl CogTileCache {
    pub fn new(cogs: Vec<Vec<cogbuilder::CogBuilder>>) -> Self {
        Self { cache: Mutex::new(Cache::new(128)), condvars: (0..256).map(|_|Condvar::new()).collect(), cogs }
    }

    pub(crate) fn get(
        &self,
        layer: u8,
        face: u8,
        level: u8,
        tile: u32,
    ) -> Result<Arc<Option<Vec<u8>>>, Error> {
        let key = (layer, face, level, tile);

        let mut cache = self.cache.lock().unwrap();
        if let Some(sector) = cache.get(key) {
            return Ok(sector);
        }

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        let hash = hasher.finish() as usize;

        if cache.is_pending(&key) {
            cache = self.condvars[hash % 256].wait_while(cache, |c| c.is_pending(&key)).unwrap();
            if let Some(tile) = cache.get(key) {
                return Ok(tile);
            }
        }

        cache.mark_pending(key);
        drop(cache);

        let uncompressed = match self.cogs[layer as usize][face as usize].read_tile(level as u32, tile)? {
            Some(bytes) => Some(cogbuilder::decompress_tile(&bytes)?),
            None => None,
        };
        let arc = Arc::new(uncompressed);

        let mut cache = self.cache.lock().unwrap();
        cache.insert(key, arc.clone());


        self.condvars[hash % 256].notify_all();
        Ok(arc)
    }

    pub fn tiles_across(&self, layer: u8, level: u32) -> u32 {
        self.cogs[layer as usize][0].tiles_across(level)
    }
}
