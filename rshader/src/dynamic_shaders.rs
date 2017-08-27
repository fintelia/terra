use std::error::Error;
use std::fs::File;
use std::io::{self, Read};
use std::iter::Iterator;
use std::path::PathBuf;
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

fn concat_file_contents<'a, I: Iterator<Item = &'a PathBuf>>(filenames: I) -> io::Result<String> {
    let mut contents = String::new();
    for filename in filenames {
        File::open(filename)?.read_to_string(&mut contents)?;
    }
    Ok(contents)
}

pub struct Shader<R: gfx::Resources> {
    shader_set: gfx::ShaderSet<R>,
    vertex_filenames: Vec<PathBuf>,
    pixel_filenames: Vec<PathBuf>,
    last_update: Instant,
}
impl<R: gfx::Resources> Shader<R> {
    pub fn simple<F: gfx::Factory<R>>(
        factory: &mut F,
        watcher: &mut ShaderDirectoryWatcher,
        vertex_source: ShaderSource,
        pixel_source: ShaderSource,
    ) -> Result<Self, Box<Error>> {
        let vertex_filenames = vertex_source
            .filenames
            .unwrap()
            .into_iter()
            .map(|f| watcher.directory.join(f))
            .collect();
        let pixel_filenames = pixel_source
            .filenames
            .unwrap()
            .into_iter()
            .map(|f| watcher.directory.join(f))
            .collect();

        Ok(Self {
            shader_set: Self::load(factory, &vertex_filenames, &pixel_filenames)?,
            last_update: Instant::now(),
            vertex_filenames,
            pixel_filenames,
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
            let new = Self::load(factory, &self.vertex_filenames, &self.pixel_filenames);
            self.last_update = Instant::now();
            if let Ok(shader_set) = new {
                self.shader_set = shader_set;
                return true;
            }
        }
        false
    }

    fn load<F: gfx::Factory<R>>(
        factory: &mut F,
        vertex_filenames: &Vec<PathBuf>,
        pixel_filenames: &Vec<PathBuf>,
    ) -> Result<gfx::ShaderSet<R>, Box<Error>> {
        let v = create_vertex_shader(
            factory,
            concat_file_contents(vertex_filenames.iter())?.as_bytes(),
        )?;
        let f = create_pixel_shader(
            factory,
            concat_file_contents(pixel_filenames.iter())?.as_bytes(),
        )?;
        Ok(gfx::ShaderSet::Simple(v, f))
    }
}
