use failure::Error;
use bincode::{self, Infinite};
use memmap::Mmap;
use serde::Serialize;
use serde::de::DeserializeOwned;

use std::mem;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::sync::Arc;

use ::*;

pub struct MMappedAsset<H> {
    pub header: H,
    pub mapping: Mmap,
}
impl<H: 'static> Asset for MMappedAsset<H> {
    fn bytes(&self) -> usize {
        mem::size_of::<H>()
    }
}
pub(crate) trait MMappedAssetDefinitionInternal {
    fn filename(&self) -> String;
    fn load(&self, context: &mut AssetLoadContext) -> Result<Arc<Asset>, Error>;
}
pub trait MMappedAssetDefinition: 'static {
    type Header: Serialize + DeserializeOwned + 'static;

    fn filename(&self) -> String;
    fn generate<W: Write>(
        &self,
        context: &mut AssetLoadContext,
        w: W,
    ) -> Result<Self::Header, Error>;
}
impl<T: MMappedAssetDefinition> MMappedAssetDefinitionInternal for T {
    fn filename(&self) -> String {
        self.filename()
    }
    fn load(&self, context: &mut AssetLoadContext) -> Result<Arc<Asset>, Error> {
        context.increment_level(&format!("Loading {}... ", &self.filename()), 100);
        let ret = (|| {
            let header_filename = context.directory().join(self.filename() + ".hdr");
            let data_filename = context.directory().join(self.filename() + ".data");

            if let (Ok(mut header), Ok(data)) =
                (File::open(&header_filename), File::open(&data_filename))
            {
                let mut contents = Vec::new();
                header.read_to_end(&mut contents)?;
                let header: T::Header = bincode::deserialize(&contents)?;
                let mapping = Mmap::open(&data, ::memmap::Protection::Read)?;
                Ok(Arc::new(MMappedAsset { header, mapping }) as Arc<Asset>)
            } else {
                context.reset(&format!("Generating {}... ", &self.filename()), 100);
                if let Some(parent) = data_filename.parent() {
                    fs::create_dir_all(parent)?;
                }
                let mut data_file = File::create(&data_filename)?;
                let header = self.generate(context, BufWriter::new(&mut data_file))?;
                context.reset(&format!("Saving {}... ", &self.filename()), 100);
                data_file.sync_all()?;

                let mut header_file = File::create(&header_filename)?;
                {
                    let mut writer = BufWriter::new(&mut header_file);
                    bincode::serialize_into(&mut writer, &header, Infinite)?;
                }
                header_file.sync_all()?;

                // Open for reading this time
                context.increment_level(&format!("Loading {}... ", &self.filename()), 100);
                let data_file = File::open(&data_filename)?;
                let mapping = Mmap::open(&data_file, ::memmap::Protection::Read)?;
                Ok(Arc::new(MMappedAsset { header, mapping }) as Arc<Asset>)
            }
        })();
        context.decrement_level();
        ret
    }
}
