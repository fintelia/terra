#![feature(nll)]

extern crate bincode;
extern crate curl;
extern crate failure;
extern crate lru_cache;
extern crate memmap;
extern crate num;
extern crate pbr;
extern crate serde;

#[cfg(test)]
#[macro_use]
extern crate serde_derive;

use failure::Error;
use lru_cache::LruCache;
use num::ToPrimitive;
use pbr::{MultiBar, Pipe, ProgressBar, Units};

use std::thread;
use std::collections::HashMap;
use std::sync::{Arc, Weak};
use std::io::Stdout;
use std::path::{Path, PathBuf};

mod asset;
mod generated;
mod mmapped;
mod web;

pub use asset::{Asset, AssetDefinition};
pub use generated::GeneratedAssetDefinition;
pub use mmapped::{MMappedAsset, MMappedAssetDefinition};
pub use web::WebAssetDefinition;

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct AssetId(usize);

pub struct AssetLoadContext<'a> {
    cache: &'a mut AssetCache,
    bars: Vec<ProgressBar<Pipe>>,
    level: usize,
}
impl<'a> AssetLoadContext<'a> {
    fn new(cache: &'a mut AssetCache) -> Self {
        let mut mb = MultiBar::<Stdout>::new();
        let mut bars = Vec::new();
        for i in 0..8 {
            let mut b = mb.create_bar(100);
            b.is_visible = i == 0;
            b.message(&format!("Level {}... ", i + 1));
            b.format("[=> ]");
            b.show_time_left = false;
            b.show_speed = false;
            b.tick();
            bars.push(b);
        }

        thread::spawn(move || {
            mb.listen();
        });

        Self {
            cache,
            bars,
            level: 0,
        }
    }

    fn directory(&self) -> &Path {
        &self.cache.directory
    }

    pub fn set_progress<N: ToPrimitive>(&mut self, value: N) {
        self.bars[self.level - 1].set(value.to_u64().unwrap());
    }

    pub fn set_progress_and_total<N: ToPrimitive, M: ToPrimitive>(&mut self, value: N, total: M) {
        self.bars[self.level - 1].total = total.to_u64().unwrap();
        self.bars[self.level - 1].set(value.to_u64().unwrap());
    }

    pub fn reset<N: ToPrimitive>(&mut self, message: &str, total: N) {
        self.bytes_display_enabled(false);
        self.bars[self.level - 1].total = total.to_u64().unwrap();
        self.bars[self.level - 1].message(message);
        self.bars[self.level - 1].set(0);
    }

    pub fn bytes_display_enabled(&mut self, enabled: bool) {
        self.bars[self.level - 1].set_units(if enabled {
            Units::Bytes
        } else {
            Units::Default
        });
    }

    pub(crate) fn increment_level<N: ToPrimitive>(&mut self, message: &str, total: N) {
        self.level += 1;
        self.bars[self.level - 1].is_visible = true;
        self.reset(message, total);
        assert!(self.level <= self.bars.len());
    }

    pub(crate) fn decrement_level(&mut self) {
        assert!(self.level > 0);
        self.bytes_display_enabled(false);
        self.bars[self.level - 1].is_visible = false;
        self.bars[self.level - 1].tick();
        self.level -= 1;
    }

    pub fn get_id(&mut self, def: AssetDefinition) -> AssetId {
        self.cache.get_id(def)
    }

    pub fn get_asset(&mut self, id: AssetId) -> Result<Arc<Asset>, Error> {
        self.cache.get_asset(id)
    }
}

pub struct AssetCache {
    assets: Vec<(Arc<AssetDefinition>, Option<Weak<Asset>>)>,
    cache: LruCache<AssetId, Arc<Asset>>,

    ids: HashMap<String, AssetId>,
    directory: PathBuf,
}
impl AssetCache {
    pub fn new(cache_directory: PathBuf, capacity: usize) -> Self {
        Self {
            assets: Vec::new(),
            cache: LruCache::new(capacity),
            ids: HashMap::new(),
            directory: cache_directory,
        }
    }

    pub fn get_id(&mut self, def: AssetDefinition) -> AssetId {
        let ids = &mut self.ids;
        let assets = &mut self.assets;

        *ids.entry(def.filename().to_owned()).or_insert_with(|| {
            let id = AssetId(assets.len());
            assets.push((Arc::from(def), None));
            id
        })
    }

    pub fn get_definition(&self, id: AssetId) -> Option<&Arc<AssetDefinition>> {
        self.assets.get(id.0).map(|a| &a.0)
    }

    pub fn get_asset(&mut self, id: AssetId) -> Result<Arc<Asset>, Error> {
        if let Some(w) = self.assets[id.0].1.as_ref().and_then(|w| w.upgrade()) {
            return Ok(w);
        }

        let a = self.assets[id.0]
            .0
            .clone()
            .load(&mut AssetLoadContext::new(self))?;
        self.cache.insert(id, a.clone());
        self.assets[id.0].1 = Some(Arc::downgrade(&a));
        Ok(a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_works() {
        #[derive(Copy, Clone, Serialize, Deserialize)]
        struct Foo;
        impl Asset for Foo {
            fn bytes(&self) -> usize {
                unimplemented!()
            }
        }
        impl GeneratedAssetDefinition for Foo {
            type Type = Foo;
            fn filename(&self) -> String {
                "".to_owned()
            }
            fn generate(&self, _context: &mut AssetLoadContext) -> Result<Self::Type, Error> {
                Err(failure::err_msg(""))
            }
        }

        let mut cache = AssetCache::new("".to_owned().into(), 128);

        let id = cache.get_id(AssetDefinition::from_generated(Foo));
        let _a = cache.get_asset(id);
    }
}
