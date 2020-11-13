use std::fs::{self, File};
use std::io::{BufWriter, Cursor, Read, Stdout, Write};
use std::ops::Drop;
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Error;
use bincode;
use dirs;
use memmap::MmapMut;
use num::ToPrimitive;
use pbr::{MultiBar, Pipe, ProgressBar, Units};
use serde::de::DeserializeOwned;
use serde::Serialize;

lazy_static! {
    pub(crate) static ref TERRA_DIRECTORY: PathBuf =
        dirs::cache_dir().unwrap_or(PathBuf::from(".")).join("terra");
}

pub(crate) struct AssetLoadContextBuf {
    bars: Vec<ProgressBar<Pipe>>,
}
impl AssetLoadContextBuf {
    pub fn new() -> Self {
        let mut mb = MultiBar::<Stdout>::new();
        let mut bars = Vec::new();
        for _ in 0..8 {
            let mut b = mb.create_bar(100);
            b.is_visible = false;
            b.format("[=> ]");
            b.show_time_left = false;
            b.show_speed = false;
            b.tick();
            bars.push(b);
        }

        thread::spawn(move || {
            mb.listen();
        });

        Self { bars }
    }
    pub fn context<N: ToPrimitive>(&mut self, message: &str, total: N) -> AssetLoadContext {
        self.bars[0].message(message);
        self.bars[0].total = total.to_u64().unwrap();
        self.bars[0].set(0);
        self.bars[0].set_units(Units::Default);
        self.bars[0].is_visible = true;
        AssetLoadContext { bars: &mut self.bars[..] }
    }
}

pub(crate) struct AssetLoadContext<'a> {
    bars: &'a mut [ProgressBar<Pipe>],
}
impl<'a> AssetLoadContext<'a> {
    pub fn set_progress<N: ToPrimitive>(&mut self, value: N) {
        self.bars[0].set(value.to_u64().unwrap());
    }

    pub fn set_progress_and_total<N: ToPrimitive, M: ToPrimitive>(&mut self, value: N, total: M) {
        self.bars[0].total = total.to_u64().unwrap();
        self.bars[0].set(value.to_u64().unwrap());
    }

    pub fn reset<N: ToPrimitive>(&mut self, message: &str, total: N) {
        self.bytes_display_enabled(false);
        self.bars[0].total = total.to_u64().unwrap();
        self.bars[0].message(message);
        self.bars[0].set(0);
    }

    pub fn bytes_display_enabled(&mut self, enabled: bool) {
        self.bars[0].set_units(if enabled { Units::Bytes } else { Units::Default });
    }

    pub fn increment_level<'b, N: ToPrimitive>(
        &'b mut self,
        message: &str,
        total: N,
    ) -> AssetLoadContext<'b>
    where
        'a: 'b,
    {
        self.bars[1].total = total.to_u64().unwrap();
        self.bars[1].message(message);
        self.bars[1].set(0);
        self.bars[1].set_units(Units::Default);
        self.bars[1].is_visible = true;
        AssetLoadContext { bars: &mut self.bars[1..] }
    }
}
impl<'a> Drop for AssetLoadContext<'a> {
    fn drop(&mut self) {
        self.bars[0].is_visible = false;
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

pub(crate) enum CompressionType {
    None,
    Snappy,
    Lz4,
}

pub(crate) trait WebAsset {
    type Type;

    fn url(&self) -> String;
    fn filename(&self) -> String;
    fn parse(&self, context: &mut AssetLoadContext, data: Vec<u8>) -> Result<Self::Type, Error>;

    fn compression(&self) -> CompressionType {
        CompressionType::None
    }
    fn credentials(&self) -> Option<(String, String)> {
        None
    }

    fn load(&self, context: &mut AssetLoadContext) -> Result<Self::Type, Error> {
        let context =
            &mut context.increment_level(&format!("Loading {}... ", &self.filename()), 100);
        let filename = TERRA_DIRECTORY.join(self.filename());

        if let Ok(file) = File::open(&filename) {
            if let Ok(mut data) = read_file(context, file) {
                match self.compression() {
                    CompressionType::Snappy => {
                        context.reset(&format!("Decompressing {}... ", &self.filename()), 100);
                        let mut uncompressed = Vec::new();
                        snap::read::FrameDecoder::new(Cursor::new(data))
                            .read_to_end(&mut uncompressed)?;
                        data = uncompressed;
                    }
                    CompressionType::Lz4 => {
                        context.reset(&format!("Decompressing {}... ", &self.filename()), 100);
                        let mut uncompressed = Vec::new();
                        lz4::Decoder::new(Cursor::new(data))?.read_to_end(&mut uncompressed)?;
                        data = uncompressed;
                    }
                    CompressionType::None => {}
                }
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
            easy.progress(true)?;
            easy.follow_location(true)?;
            easy.fail_on_error(true)?;
            if let Some((username, password)) = self.credentials() {
                easy.cookie_file("")?;
                easy.unrestricted_auth(true)?;
                easy.username(&username)?;
                easy.password(&password)?;
            }
            let mut transfer = easy.transfer();
            transfer.write_function(|d| {
                let len = d.len();
                data.extend(d);
                Ok(len)
            })?;
            transfer.progress_function(|t, c, _, _| {
                if t > 0.0 {
                    context.set_progress_and_total(c, t);
                }
                true
            })?;
            transfer.perform()?;
        }

        context.reset(&format!("Saving {}... ", &self.filename()), 100);
        if let Some(parent) = filename.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = File::create(&filename)?;
        match self.compression() {
            CompressionType::Snappy => {
                snap::write::FrameEncoder::new(&mut file).write_all(&data)?
            }
            CompressionType::Lz4 => {
                let mut writer = lz4::EncoderBuilder::new().level(9).build(&mut file)?;
                writer.write_all(&data)?;
                writer.finish().1?
            }
            CompressionType::None => file.write_all(&data)?,
        }
        file.sync_all()?;
        context.reset(&format!("Parsing {}... ", &self.filename()), 100);
        Ok(self.parse(context, data)?)
    }
}

pub(crate) trait GeneratedAsset {
    type Type: Serialize + DeserializeOwned;

    fn filename(&self) -> String;
    fn generate(&self, context: &mut AssetLoadContext) -> Result<Self::Type, Error>;

    fn load(&self, context: &mut AssetLoadContext) -> Result<Self::Type, Error> {
        let context =
            &mut context.increment_level(&format!("Loading {}... ", &self.filename()), 100);
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
                bincode::serialize_into(&mut writer, &generated)?;
            }
            file.sync_all()?;
            Ok(generated)
        }
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

    fn load(&self, context: &mut AssetLoadContext) -> Result<(Self::Header, MmapMut), Error> {
        let context =
            &mut context.increment_level(&format!("Loading {}... ", &self.filename()), 100);
        let header_filename = TERRA_DIRECTORY.join(self.filename() + ".hdr");
        let data_filename = TERRA_DIRECTORY.join(self.filename() + ".data");

        if let (Ok(mut header), Ok(data)) = (
            File::open(&header_filename),
            File::with_options().read(true).write(true).open(&data_filename),
        ) {
            let mut contents = Vec::new();
            header.read_to_end(&mut contents)?;
            let header = bincode::deserialize(&contents)?;
            let mapping = unsafe { MmapMut::map_mut(&data)? };
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
                bincode::serialize_into(&mut writer, &header)?;
            }
            header_file.sync_all()?;

            // Open for reading this time
            context.reset(&format!("Loading {}... ", &self.filename()), 100);
            let data_file = File::with_options().read(true).write(true).open(&data_filename)?;
            let mapping = unsafe { MmapMut::map_mut(&data_file)? };
            Ok((header, mapping))
        }
    }
}

impl<H: Serialize + DeserializeOwned, A: WebAsset<Type = (H, Vec<u8>)>> MMappedAsset for A {
    type Header = H;

    fn filename(&self) -> String {
        WebAsset::filename(self)
    }

    fn generate<W: Write>(
        &self,
        context: &mut AssetLoadContext,
        mut w: W,
    ) -> Result<Self::Header, Error> {
        let (header, data) = WebAsset::load(self, context)?;
        w.write_all(&data[..])?;
        Ok(header)
    }
}
