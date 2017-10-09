use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::path::PathBuf;
use std::io::{BufWriter, Read, Write};

use bincode::{self, Infinite};
use memmap::Mmap;
use serde::Serialize;
use serde::de::DeserializeOwned;

lazy_static! {
    static ref TERRA_DIRECTORY: PathBuf = {
        env::home_dir().unwrap_or(PathBuf::from(".")).join(".terra")
    };
}

pub trait WebAsset {
    type Type;

    fn url(&self) -> String;
    fn filename(&self) -> String;
    fn parse(&self, data: Vec<u8>) -> Result<Self::Type, Box<Error>>;

    fn load(&self) -> Result<Self::Type, Box<Error>> {
        let filename = TERRA_DIRECTORY.join(self.filename());
        let mut data = Vec::<u8>::new();
        if let Ok(mut file) = File::open(&filename) {
            file.read_to_end(&mut data)?;
        } else {
            {
                use curl::easy::Easy;
                let mut easy = Easy::new();
                easy.url(&self.url())?;
                let mut easy = easy.transfer();
                easy.write_function(|d| {
                    let len = d.len();
                    data.extend(d);
                    Ok(len)
                })?;
                easy.perform()?;
            }

            if let Some(parent) = filename.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut file = File::create(filename)?;
            file.write_all(&data)?;
            file.sync_all()?;
        }
        Ok(self.parse(data)?)
    }
}

pub trait GeneratedAsset {
    type Type: Serialize + DeserializeOwned;

    fn filename(&self) -> String;
    fn generate(&self) -> Result<Self::Type, Box<Error>>;

    fn load(&self) -> Result<Self::Type, Box<Error>> {
        let filename = TERRA_DIRECTORY.join(self.filename());
        if let Ok(mut file) = File::open(&filename) {
            let mut contents = Vec::new();
            file.read_to_end(&mut contents)?;
            Ok(bincode::deserialize(&contents)?)
        } else {
            let generated = self.generate()?;
            if let Some(parent) = filename.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut file = File::create(&filename)?;
            {
                let mut writer = BufWriter::new(&mut file);
                bincode::serialize_into(&mut writer, &generated, Infinite)?;
            }
            file.sync_all()?;
            Ok(generated)
        }
    }
}

pub(crate) trait MMappedAsset {
    type Header: Serialize + DeserializeOwned;

    fn filename(&self) -> String;
    fn generate<W: Write>(&self, w: W) -> Result<Self::Header, Box<Error>>;

    fn load(&self) -> Result<(Self::Header, Mmap), Box<Error>> {
        let header_filename = TERRA_DIRECTORY.join(self.filename() + ".hdr");
        let data_filename = TERRA_DIRECTORY.join(self.filename() + ".data");

        if let (Ok(mut header), Ok(data)) =
            (File::open(&header_filename), File::open(&data_filename))
        {
            let mut contents = Vec::new();
            header.read_to_end(&mut contents)?;
            let header = bincode::deserialize(&contents)?;
            let mapping = Mmap::open(&data, ::memmap::Protection::Read)?;
            Ok((header, mapping))
        } else {
            if let Some(parent) = data_filename.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut data_file = File::create(&data_filename)?;
            let header = self.generate(BufWriter::new(&mut data_file))?;
            data_file.sync_all()?;

            let mut header_file = File::create(&header_filename)?;
            {
                let mut writer = BufWriter::new(&mut header_file);
                bincode::serialize_into(&mut writer, &header, Infinite)?;
            }
            header_file.sync_all()?;

            let mapping = Mmap::open(&data_file, ::memmap::Protection::Read)?;
            Ok((header, mapping))
        }
    }
}
