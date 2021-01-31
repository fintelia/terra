use anyhow::Error;
use bincode;
use dirs;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use memmap::MmapMut;
use num::ToPrimitive;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::io::{BufWriter, Cursor, Read, Write};
use std::ops::Drop;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use std::{
    fs::{self, File, OpenOptions},
    sync::Arc,
};

lazy_static! {
    pub(crate) static ref TERRA_DIRECTORY: PathBuf =
        dirs::cache_dir().unwrap_or(PathBuf::from(".")).join("terra");
    static ref PROGRESS_BAR_STYLE: ProgressStyle = ProgressStyle::default_bar()
        .template("{msg} {pos}/{len} [{wide_bar}] {percent}% {per_sec} {eta}")
        .progress_chars("=> ");
    static ref PROGRESS_BAR_STYLE_BYTES: ProgressStyle = ProgressStyle::default_bar()
        .template("{msg} {bytes}/{total_bytes} [{wide_bar}] {percent}% {per_sec} {eta}")
        .progress_chars("=> ");
}

pub(crate) struct AssetLoadContextBuf {
    bars: Arc<MultiProgress>,
}
impl AssetLoadContextBuf {
    pub fn new() -> Self {
        Self { bars: Arc::new(MultiProgress::new()) }
    }
    pub fn context<N: ToPrimitive>(&mut self, message: &str, total: N) -> AssetLoadContext {
        let bar = ProgressBar::new(total.to_u64().unwrap());
        let bar = self.bars.add(bar);
        bar.set_style(PROGRESS_BAR_STYLE.clone());
        bar.set_message(message);

        let multibar = Arc::clone(&self.bars);
        std::thread::spawn(move || multibar.join_and_clear());

        AssetLoadContext { inner: self, bar }
    }
}

pub(crate) struct AssetLoadContext<'a> {
    inner: &'a mut AssetLoadContextBuf,
    bar: ProgressBar,
}
impl<'a> AssetLoadContext<'a> {
    pub fn set_progress<N: ToPrimitive>(&mut self, value: N) {
        self.bar.set_position(value.to_u64().unwrap());
    }

    pub fn set_progress_and_total<N: ToPrimitive, M: ToPrimitive>(&mut self, value: N, total: M) {
        self.bar.set_position(value.to_u64().unwrap());
        self.bar.set_length(total.to_u64().unwrap());
    }

    pub fn reset<N: ToPrimitive>(&mut self, message: &str, total: N) {
        let new_bar = ProgressBar::new(total.to_u64().unwrap());
        let new_bar = self.inner.bars.add(new_bar);

        self.bar.finish_and_clear();

        self.bar = new_bar;
        self.bar.set_style(PROGRESS_BAR_STYLE.clone());
        self.bar.set_message(message);
    }

    pub fn bytes_display_enabled(&mut self, enabled: bool) {
        if enabled {
            self.bar.set_style(PROGRESS_BAR_STYLE.clone());
        } else {
            self.bar.set_style(PROGRESS_BAR_STYLE_BYTES.clone());
        }
        self.bar.tick();
    }

    pub fn increment_level<'b, N: ToPrimitive>(
        &'b mut self,
        message: &str,
        total: N,
    ) -> AssetLoadContext<'b>
    where
        'a: 'b,
    {
        let bar = ProgressBar::new(total.to_u64().unwrap());
        let bar = self.inner.bars.add(bar);
        bar.set_style(PROGRESS_BAR_STYLE.clone());
        bar.set_message(message);

        AssetLoadContext { inner: self.inner, bar }
    }
}
impl<'a> Drop for AssetLoadContext<'a> {
    fn drop(&mut self) {
        self.bar.finish_and_clear();
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
    #[allow(unused)]
    Snappy,
    #[allow(unused)]
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
            OpenOptions::new().read(true).write(true).open(&data_filename),
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
            let data_file = OpenOptions::new().read(true).write(true).open(&data_filename)?;
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
