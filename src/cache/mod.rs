use std::fs::{self, File};
use std::io::{BufWriter, Read, Stdout, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use std::thread;

use bincode::{self, Infinite};
use dirs;
use failure::Error;
use memmap::Mmap;
use num::ToPrimitive;
use pbr::{MultiBar, Pipe, ProgressBar, Units};
use serde::Serialize;
use serde::de::DeserializeOwned;

lazy_static! {
    static ref TERRA_DIRECTORY: PathBuf =
        { dirs::cache_dir().unwrap_or(PathBuf::from(".")).join("terra") };
}

pub(crate) struct AssetLoadContext {
    bars: Vec<ProgressBar<Pipe>>,
    level: usize,
}
impl AssetLoadContext {
    pub fn new() -> Self {
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

        Self { bars, level: 0 }
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

    pub fn increment_level<N: ToPrimitive>(&mut self, message: &str, total: N) {
        self.level += 1;
        self.bars[self.level - 1].is_visible = true;
        self.reset(message, total);
        assert!(self.level <= self.bars.len());
    }

    pub fn decrement_level(&mut self) {
        assert!(self.level > 0);
        self.bytes_display_enabled(false);
        self.bars[self.level - 1].is_visible = false;
        self.bars[self.level - 1].tick();
        self.level -= 1;
    }
}

fn read_file(context: &mut AssetLoadContext, mut file: File) -> Result<Vec<u8>, Error> {
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

pub(crate) trait WebAsset {
    type Type;

    fn url(&self) -> String;
    fn filename(&self) -> String;
    fn parse(&self, context: &mut AssetLoadContext, data: Vec<u8>) -> Result<Self::Type, Error>;

    fn credentials(&self) -> Option<(String, String)> {
        None
    }

    fn load(&self, context: &mut AssetLoadContext) -> Result<Self::Type, Error> {
        context.increment_level(&format!("Loading {}... ", &self.filename()), 100);
        let ret = (|| {
            let filename = TERRA_DIRECTORY.join(self.filename());

            if let Ok(file) = File::open(&filename) {
                if let Ok(data) = read_file(context, file) {
                    context.reset(&format!("Parsing {}... ", &self.filename()), 100);
                    if let Ok(asset) = self.parse(context, data) {
                        return Ok(asset);
                    }
                }
            }

            let mut data = Vec::<u8>::new();
            {
                context.reset(&format!("Downloading {}... ", &self.filename()), 100);
                // Bytes display will be disabled by the reset() below, or in the event of an error,
                // by the decrement_level() call in the outer scope.
                context.bytes_display_enabled(true);

                use curl::easy::Easy;
                let mut easy = Easy::new();
                easy.url(&self.url())?;
                easy.follow_location(true)?;
                if let Some((username, password)) = self.credentials() {
                    easy.cookie_file("")?;
                    easy.unrestricted_auth(true)?;
                    easy.username(&username)?;
                    easy.password(&password)?;
                }
                let mut easy = easy.transfer();
                easy.write_function(|d| {
                    let len = d.len();
                    data.extend(d);
                    Ok(len)
                })?;
                easy.progress_function(|c, t, _, _| {
                    if t > 0.0 {
                        context.set_progress_and_total(c, t);
                    }
                    true
                })?;
                easy.perform()?;
            }

            context.reset(&format!("Saving {}... ", &self.filename()), 100);
            if let Some(parent) = filename.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut file = File::create(&filename)?;
            file.write_all(&data)?;
            file.sync_all()?;
            context.reset(&format!("Parsing {}... ", &self.filename()), 100);
            Ok(self.parse(context, data)?)
        })();
        context.decrement_level();
        ret
    }
}

pub(crate) trait GeneratedAsset {
    type Type: Serialize + DeserializeOwned;

    fn filename(&self) -> String;
    fn generate(&self, context: &mut AssetLoadContext) -> Result<Self::Type, Error>;

    fn load(&self, context: &mut AssetLoadContext) -> Result<Self::Type, Error> {
        context.increment_level(&format!("Loading {}... ", &self.filename()), 100);
        let ret = (|| {
            let filename = TERRA_DIRECTORY.join(self.filename());
            if let Ok(file) = File::open(&filename) {
                Ok(bincode::deserialize(&read_file(context, file)?)?)
            } else {
                context.reset(&format!("Generating {}... ", &self.filename()), 100);
                let generated = self.generate(context)?;
                context.reset(&format!("Saving {}... ", &self.filename()), 100);
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
        })();
        context.decrement_level();
        ret
    }
}

pub(crate) trait MMappedAsset {
    type Header: Serialize + DeserializeOwned;

    fn filename(&self) -> String;
    fn generate<W: Write>(
        &self,
        context: &mut AssetLoadContext,
        w: W,
    ) -> Result<Self::Header, Error>;

    fn load(&self, context: &mut AssetLoadContext) -> Result<(Self::Header, Mmap), Error> {
        context.increment_level(&format!("Loading {}... ", &self.filename()), 100);
        let ret = (|| {
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
                Ok((header, mapping))
            }
        })();
        context.decrement_level();
        ret
    }
}
