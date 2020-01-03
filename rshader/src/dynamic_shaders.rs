use std::collections::HashMap;
use std::fs::{self, File};
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

    last_modifications: HashMap<PathBuf, Instant>,
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

        Ok(Self { directory, _watcher: watcher, watcher_rx, last_modifications: HashMap::new() })
    }

    fn detect_changes(&mut self) {
        while let Ok(event) = self.watcher_rx.try_recv() {
            if let DebouncedEvent::Write(p) = event {
                self.last_modifications.insert(p, Instant::now());
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
    vertex: Option<Vec<u8>>,
    fragment: Option<Vec<u8>>,
    compute: Option<Vec<u8>>,

    vertex_filenames: Vec<PathBuf>,
    fragment_filenames: Vec<PathBuf>,
    compute_filenames: Vec<PathBuf>,
    last_update: Instant,
}
impl ShaderSet {
    pub fn simple(
        watcher: &mut ShaderDirectoryWatcher,
        vertex_source: ShaderSource,
        fragment_source: ShaderSource,
    ) -> Result<Self, failure::Error> {
        let vertex_filenames = vertex_source
            .filenames
            .unwrap()
            .into_iter()
            .map(|f| fs::canonicalize(watcher.directory.join(f)).unwrap())
            .collect::<Vec<_>>();
        let fragment_filenames = fragment_source
            .filenames
            .unwrap()
            .into_iter()
            .map(|f| fs::canonicalize(watcher.directory.join(f)).unwrap())
            .collect::<Vec<_>>();
        let vertex = create_vertex_shader(&concat_file_contents(vertex_filenames.iter())?)?;
        let fragment = create_fragment_shader(&concat_file_contents(fragment_filenames.iter())?)?;

        Ok(Self {
            vertex: Some(vertex),
            fragment: Some(fragment),
            compute: None,
            last_update: Instant::now(),
            vertex_filenames: vertex_filenames,
            fragment_filenames: fragment_filenames,
            compute_filenames: Vec::new(),
        })
    }

    pub fn compute_only(
        watcher: &mut ShaderDirectoryWatcher,
        compute_source: ShaderSource,
    ) -> Result<Self, failure::Error> {
        let compute_filenames = compute_source
            .filenames
            .unwrap()
            .into_iter()
            .map(|f| fs::canonicalize(watcher.directory.join(f)).unwrap())
            .collect::<Vec<_>>();
        let compute = create_compute_shader(&concat_file_contents(compute_filenames.iter())?)?;

        Ok(Self {
            vertex: None,
            fragment: None,
            compute: Some(compute),
            last_update: Instant::now(),
            vertex_filenames: Vec::new(),
            fragment_filenames: Vec::new(),
            compute_filenames: compute_filenames,
        })
    }

    /// Refreshes the shader if necessary. Returns whether a refresh happened.
    pub fn refresh(&mut self, directory_watcher: &mut ShaderDirectoryWatcher) -> bool {
        directory_watcher.detect_changes();

        let needs_update = self
            .vertex_filenames
            .iter()
            .chain(self.fragment_filenames.iter())
            .chain(self.compute_filenames.iter())
            .filter_map(|n| directory_watcher.last_modifications.get(n))
            .any(|&t| t > self.last_update);

        if needs_update {
            self.last_update = Instant::now();

            let new_shaders = || -> Result<_, failure::Error> {
                let (mut vs, mut fs, mut cs) = (None, None, None);
                if !self.vertex_filenames.is_empty() {
                    vs = Some(create_vertex_shader(&concat_file_contents(
                        self.vertex_filenames.iter(),
                    )?)?);
                }
                if !self.fragment_filenames.is_empty() {
                    fs = Some(create_fragment_shader(&concat_file_contents(
                        self.fragment_filenames.iter(),
                    )?)?);
                }
                if !self.compute_filenames.is_empty() {
                    cs = Some(create_compute_shader(&concat_file_contents(
                        self.compute_filenames.iter(),
                    )?)?);
                }
                Ok((vs, fs, cs))
            }();

            if let Ok((vs, fs, cs)) = new_shaders {
                self.vertex = vs;
                self.fragment = fs;
                self.compute = cs;
                return true;
            }
        }
        false
    }

    pub fn vertex(&self) -> &[u8] {
        self.vertex.as_ref().unwrap()
    }
    pub fn fragment(&self) -> &[u8] {
        self.fragment.as_ref().unwrap()
    }
    pub fn compute(&self) -> &[u8] {
        self.compute.as_ref().unwrap()
    }
}
