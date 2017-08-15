use std::error::Error;
use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

use gfx;
use notify::{self, DebouncedEvent, RecommendedWatcher, Watcher, RecursiveMode};

use super::*;

pub struct ShaderDirectoryWatcher {
    directory: PathBuf,
    _watcher: RecommendedWatcher,
    watcher_rx: Receiver<DebouncedEvent>,

    last_modification: Instant,
}
impl ShaderDirectoryWatcher {
    pub fn new<P>(directory: P) -> Result<Self, notify::Error>
    where
        PathBuf: From<P>,
    {
        let directory = PathBuf::from(directory);
        let (tx, watcher_rx) = mpsc::channel();
        let mut watcher = notify::watcher(tx, Duration::from_millis(50))?;
        watcher.watch(&directory, RecursiveMode::Recursive)?;

        Ok(Self {
            directory,
            _watcher: watcher,
            watcher_rx,
            last_modification: Instant::now(),
        })
    }

    fn detect_changes(&mut self) {
        while let Ok(event) = self.watcher_rx.try_recv() {
            if let DebouncedEvent::Write(_) = event {
                self.last_modification = Instant::now();
            }
        }
    }
}

fn file_contents(filename: &Path) -> io::Result<String> {
    let mut file = File::open(filename)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
}

pub struct Shader<R: gfx::Resources> {
    shader_set: gfx::ShaderSet<R>,
    vertex_filename: PathBuf,
    pixel_filename: PathBuf,
    last_update: Instant,
}
impl<R: gfx::Resources> Shader<R> {
    pub fn simple<F: gfx::Factory<R>>(
        factory: &mut F,
        watcher: &mut ShaderDirectoryWatcher,
        vertex_source: ShaderSource,
        pixel_source: ShaderSource,
    ) -> Result<Self, Box<Error>> {
        let vertex_filename = watcher.directory.join(vertex_source.filename.unwrap());
        let pixel_filename = watcher.directory.join(pixel_source.filename.unwrap());

        Ok(Self {
            shader_set: Self::load(factory, &vertex_filename, &pixel_filename)?,
            last_update: Instant::now(),
            vertex_filename,
            pixel_filename,
        })
    }

    pub fn as_shader_set(&self) -> &gfx::ShaderSet<R> {
        &self.shader_set
    }

    /// Refreshes the shader if necessary. Returns whether a refresh happened.
    pub fn refresh<F: gfx::Factory<R>>(
        &mut self,
        factory: &mut F,
        directory_watcher: &mut ShaderDirectoryWatcher,
    ) -> bool {
        directory_watcher.detect_changes();
        if directory_watcher.last_modification > self.last_update {
            let new = Self::load(factory, &self.vertex_filename, &self.pixel_filename);
            if let Ok(shader_set) = new {
                self.shader_set = shader_set;
                return true;
            }
        }
        false
    }

    fn load<F: gfx::Factory<R>>(
        factory: &mut F,
        vertex_filename: &Path,
        pixel_filename: &Path,
    ) -> Result<gfx::ShaderSet<R>, Box<Error>> {
        let v = create_vertex_shader(factory, file_contents(vertex_filename)?.as_bytes())?;
        let f = create_pixel_shader(factory, file_contents(pixel_filename)?.as_bytes())?;
        Ok(gfx::ShaderSet::Simple(v, f))
    }
}
