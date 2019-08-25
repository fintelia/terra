use std::fs::File;
use std::io::{self, Read};
use std::iter::Iterator;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

use notify::{self, DebouncedEvent, RecommendedWatcher, RecursiveMode, Watcher};

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

pub struct ShaderSet {
    vertex: SpirvShader,
    fragment: SpirvShader,

    vertex_filenames: Vec<PathBuf>,
    pixel_filenames: Vec<PathBuf>,
    last_update: Instant,
}
impl ShaderSet {
    pub fn simple(
        watcher: &mut ShaderDirectoryWatcher,
        vertex_source: ShaderSource,
        pixel_source: ShaderSource,
    ) -> Result<Self, failure::Error> {
        let vertex_filenames: Vec<_> = vertex_source
            .filenames
            .unwrap()
            .into_iter()
            .map(|f| watcher.directory.join(f))
            .collect();
        let pixel_filenames: Vec<_> = pixel_source
            .filenames
            .unwrap()
            .into_iter()
            .map(|f| watcher.directory.join(f))
            .collect();
        let (vertex, fragment) = Self::load(&vertex_filenames, &pixel_filenames)?;

        Ok(Self {
            vertex,
            fragment,
            last_update: Instant::now(),
            vertex_filenames,
            pixel_filenames,
        })
    }

    /// Refreshes the shader if necessary. Returns whether a refresh happened.
    pub fn refresh(&mut self, directory_watcher: &mut ShaderDirectoryWatcher) -> bool {
        directory_watcher.detect_changes();
        if directory_watcher.last_modification > self.last_update {
            let new = Self::load(&self.vertex_filenames, &self.pixel_filenames);
            self.last_update = Instant::now();
            if let Ok((vertex, fragment)) = new {
                self.vertex = vertex;
                self.fragment = fragment;
                return true;
            }
        }
        false
    }

    fn load(
        vertex_filenames: &[PathBuf],
        pixel_filenames: &[PathBuf],
    ) -> Result<(SpirvShader, SpirvShader), failure::Error> {
        let v = create_vertex_shader(&concat_file_contents(vertex_filenames.iter())?)?;
        let f = create_pixel_shader(&concat_file_contents(pixel_filenames.iter())?)?;
        Ok((v, f))
    }

    pub fn vertex(&self) -> &SpirvShader {
        &self.vertex
    }
    pub fn fragment(&self) -> &SpirvShader {
        &self.fragment
    }
}
