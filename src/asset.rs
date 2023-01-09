use anyhow::Error;
use dirs;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::borrow::Cow;
use std::io::{Read, Write};
use std::ops::Drop;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use std::{
    fs::{self, File},
    sync::Arc,
};

lazy_static! {
    pub(crate) static ref TERRA_DIRECTORY: PathBuf =
        dirs::cache_dir().unwrap_or(PathBuf::from(".")).join("terra");
    static ref PROGRESS_BAR_STYLE: ProgressStyle = ProgressStyle::default_bar()
        .template("{msg} {pos}/{len} [{wide_bar}] {percent}% {per_sec} {eta}")
        .unwrap()
        .progress_chars("=> ");
    static ref PROGRESS_BAR_STYLE_BYTES: ProgressStyle = ProgressStyle::default_bar()
        .template("{msg} {bytes}/{total_bytes} [{wide_bar}] {percent}% {per_sec} {eta}")
        .unwrap()
        .progress_chars("=> ");
}

pub(crate) struct AssetLoadContextBuf {
    bars: Arc<MultiProgress>,
}
impl AssetLoadContextBuf {
    pub fn new() -> Self {
        Self { bars: Arc::new(MultiProgress::new()) }
    }
    pub fn context(
        &mut self,
        message: impl Into<Cow<'static, str>>,
        total: u64,
    ) -> AssetLoadContext {
        let bar = ProgressBar::new(total);
        let bar = self.bars.add(bar);
        bar.set_style(PROGRESS_BAR_STYLE.clone());
        bar.set_message(message);

        AssetLoadContext { inner: self, bar }
    }
}

pub(crate) struct AssetLoadContext<'a> {
    inner: &'a mut AssetLoadContextBuf,
    bar: ProgressBar,
}
impl<'a> AssetLoadContext<'a> {
    pub fn set_progress(&mut self, value: u64) {
        self.bar.set_position(value);
    }

    pub fn set_progress_and_total(&mut self, value: u64, total: u64) {
        self.bar.set_position(value);
        self.bar.set_length(total);
    }

    pub fn reset(&mut self, message: impl Into<Cow<'static, str>>, total: u64) {
        let new_bar = ProgressBar::new(total);
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

    pub fn increment_level<'b>(
        &'b mut self,
        message: impl Into<Cow<'static, str>>,
        total: u64,
    ) -> AssetLoadContext<'b>
    where
        'a: 'b,
    {
        let bar = ProgressBar::new(total);
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
        let file_len = file.metadata()?.len();
        let mut bytes_read = 0u64;
        let mut contents = vec![0; file_len as usize];
        context.set_progress_and_total(0, file_len);
        let mut last_progress_update = Instant::now();
        while bytes_read < file_len {
            let buf =
                &mut contents[bytes_read as usize..file_len.min(bytes_read + 32 * 1024) as usize];
            match file.read(buf)? {
                0 => break,
                n => bytes_read += n as u64,
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

    fn load(&self, context: &mut AssetLoadContext) -> Result<Self::Type, Error> {
        let context =
            &mut context.increment_level(format!("Loading {}... ", &self.filename()), 100);
        let filename = TERRA_DIRECTORY.join(self.filename());

        if let Ok(file) = File::open(&filename) {
            if let Ok(data) = read_file(context, file) {
                context.reset(format!("Parsing {}... ", &self.filename()), 100);
                if let Ok(asset) = self.parse(context, data) {
                    return Ok(asset);
                }
            }
        }

        // TODO: use a streaming parser to have the progress bar refresh during download
        context.reset(format!("Downloading {}... ", &self.filename()), 100);
        let data = reqwest::blocking::get(self.url())?.bytes()?.to_vec();

        context.reset(format!("Saving {}... ", &self.filename()), 100);
        if let Some(parent) = filename.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = File::create(&filename)?;
        file.write_all(&data)?;
        file.sync_all()?;
        context.reset(format!("Parsing {}... ", &self.filename()), 100);
        Ok(self.parse(context, data)?)
    }
}
