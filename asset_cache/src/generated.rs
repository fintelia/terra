use failure::Error;
use bincode::{self, Infinite};
use serde::Serialize;
use serde::de::DeserializeOwned;

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::sync::Arc;

use ::*;

pub trait GeneratedAssetDefinition: 'static {
    type Type: Serialize + DeserializeOwned + Asset;
    fn filename(&self) -> String;
    fn generate(&self, context: &mut AssetLoadContext) -> Result<Self::Type, Error>;
}
pub(crate) trait GeneratedAssetDefinitionInternal {
    fn filename(&self) -> String;
    fn load(&self, context: &mut AssetLoadContext) -> Result<Arc<Asset>, Error>;
}
impl<T: GeneratedAssetDefinition> GeneratedAssetDefinitionInternal for T {
    fn filename(&self) -> String {
        self.filename()
    }
    fn load(&self, context: &mut AssetLoadContext) -> Result<Arc<Asset>, Error> {
        context.increment_level(&format!("Loading {}... ", &self.filename()), 100);
        let ret = (|| {
            let filename = context.directory().join(self.filename());
            if let Ok(file) = File::open(&filename) {
                let t: T::Type = bincode::deserialize(&asset::read_file(context, file)?)?;
                Ok(Arc::new(t) as Arc<Asset>)
            } else {
                context.reset(&format!("Generating {}... ", &self.filename()), 100);
                let t = GeneratedAssetDefinition::generate(self, context)?;
                let bytes = bincode::serialize(&t, Infinite)?;
                let generated = Arc::new(t) as Arc<Asset>;
                context.reset(&format!("Saving {}... ", &self.filename()), 100);
                if let Some(parent) = filename.parent() {
                    fs::create_dir_all(parent)?;
                }
                let mut file = File::create(&filename)?;
                {
                    let mut writer = BufWriter::new(&mut file);
                    writer.write_all(&bytes)?;
                }
                file.sync_all()?;
                Ok(generated)
            }
        })();
        context.decrement_level();
        ret
    }
}
