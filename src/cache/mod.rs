use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::path::PathBuf;
use std::io::{Read, Write};

use bincode::{self, Infinite};
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
            bincode::serialize_into(&mut file, &generated, Infinite)?;
            Ok(generated)
        }
    }
}
