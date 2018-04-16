use failure::Error;
use serde::Serialize;
use serde::de::DeserializeOwned;

use std::fs::File;
use std::time::{Duration, Instant};
use std::io::Read;
use std::any::Any;
use std::sync::Arc;

use ::*;
use web::*;
use generated::*;
use mmapped::*;

enum AssetDefinitionInternal {
    Generated(Box<GeneratedAssetDefinitionInternal>),
    Web(Box<WebAssetDefinition>),
    MMapped(Box<MMappedAssetDefinitionInternal>),
}
pub struct AssetDefinition(AssetDefinitionInternal);
impl AssetDefinition {
    pub fn from_generated<D, T>(def: D) -> Self
    where
        D: GeneratedAssetDefinition<Type = T>,
        T: Serialize + DeserializeOwned + Asset,
    {
        AssetDefinition(AssetDefinitionInternal::Generated(Box::new(def)))
    }
    pub fn from_web<D: WebAssetDefinition>(def: D) -> Self {
        AssetDefinition(AssetDefinitionInternal::Web(Box::new(def)))
    }
    pub fn from_mmapped<D, H>(def: D) -> Self
    where
        D: MMappedAssetDefinition<Header = H>,
        H: Serialize + DeserializeOwned + Asset,
    {
        AssetDefinition(AssetDefinitionInternal::MMapped(Box::new(def)))
    }

    pub(crate) fn filename(&self) -> String {
        match self.0 {
            AssetDefinitionInternal::Generated(ref b) => b.filename(),
            AssetDefinitionInternal::Web(ref b) => b.filename(),
            AssetDefinitionInternal::MMapped(ref b) => b.filename(),
        }
    }
    pub(crate) fn load(&self, context: &mut AssetLoadContext) -> Result<Arc<Asset>, Error> {
        match self.0 {
            AssetDefinitionInternal::Generated(ref b) => b.load(context),
            AssetDefinitionInternal::Web(ref b) => b.load(context),
            AssetDefinitionInternal::MMapped(ref b) => b.load(context),
        }
    }
}

pub trait Asset: Any + 'static {
    fn bytes(&self) -> usize;
}

pub(crate) fn read_file(context: &mut AssetLoadContext, mut file: File) -> Result<Vec<u8>, Error> {
    context.bytes_display_enabled(true);
    let ret = (|| {
        let file_len = file.metadata()?.len() as usize;
        let mut bytes_read = 0;
        let mut contents = vec![0; file_len];
        context.set_progress_and_total(0, file_len);
        let mut last_progress_update = Instant::now();
        while bytes_read < file_len {
            let buf = &mut contents[bytes_read..file_len.min(bytes_read + 32 * 1024)];
            match file.read(buf)? {
                0 => break,
                n => bytes_read += n,
            }
            if last_progress_update.elapsed() > Duration::from_millis(10) {
                context.set_progress(bytes_read);
                last_progress_update = Instant::now();
            }
        }
        file.read_to_end(&mut contents)?;
        Ok(contents)
    })();
    context.bytes_display_enabled(false);
    ret
}
