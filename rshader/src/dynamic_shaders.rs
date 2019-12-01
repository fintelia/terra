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
    vertex: Option<SpirvShader>,
    fragment: Option<SpirvShader>,
    compute: Option<(SpirvShader, Vec<u8>)>,

    vertex_filenames: Option<Vec<PathBuf>>,
    fragment_filenames: Option<Vec<PathBuf>>,
    compute_filenames: Option<Vec<PathBuf>>,
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
            .map(|f| watcher.directory.join(f))
            .collect::<Vec<_>>();
        let fragment_filenames = fragment_source
            .filenames
            .unwrap()
            .into_iter()
            .map(|f| watcher.directory.join(f))
            .collect::<Vec<_>>();
        let vertex = create_vertex_shader(&concat_file_contents(vertex_filenames.iter())?)?;
        let fragment = create_fragment_shader(&concat_file_contents(fragment_filenames.iter())?)?;

        Ok(Self {
            vertex: Some(vertex),
            fragment: Some(fragment),
            compute: None,
            last_update: Instant::now(),
            vertex_filenames: Some(vertex_filenames),
            fragment_filenames: Some(fragment_filenames),
            compute_filenames: None,
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
            .map(|f| watcher.directory.join(f))
            .collect::<Vec<_>>();
        let compute = create_compute_shader(&concat_file_contents(compute_filenames.iter())?)?;

        Ok(Self {
            vertex: None,
            fragment: None,
            compute: Some(compute),
            last_update: Instant::now(),
            vertex_filenames: None,
            fragment_filenames: None,
            compute_filenames: Some(compute_filenames),
        })
    }

    /// Refreshes the shader if necessary. Returns whether a refresh happened.
    pub fn refresh(&mut self, directory_watcher: &mut ShaderDirectoryWatcher) -> bool {
        directory_watcher.detect_changes();
        if directory_watcher.last_modification > self.last_update {
            self.last_update = Instant::now();

            let new_shaders = || -> Result<_, failure::Error> {
                let (mut vs, mut fs, mut cs) = (None, None, None);
                if let Some(ref s) = self.vertex_filenames {
                    vs = Some(create_vertex_shader(&concat_file_contents(s.iter())?)?);
                }
                if let Some(ref s) = self.fragment_filenames {
                    fs = Some(create_fragment_shader(&concat_file_contents(s.iter())?)?);
                }
                if let Some(ref s) = self.compute_filenames {
                    cs = Some(create_compute_shader(&concat_file_contents(s.iter())?)?);
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

    pub fn vertex(&self) -> &SpirvShader {
        self.vertex.as_ref().unwrap()
    }
    pub fn fragment(&self) -> &SpirvShader {
        self.fragment.as_ref().unwrap()
    }
    pub fn compute(&self) -> &[u8] {
        self.compute.as_ref().map(|v| &v.1).unwrap()
    }
}
