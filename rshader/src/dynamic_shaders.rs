use notify::{self, DebouncedEvent, RecommendedWatcher, RecursiveMode, Watcher};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Read};
use std::iter::Iterator;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

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

pub struct ShaderSet {
    inner: ShaderSetInner,
    vertex_source: Option<ShaderSource>,
    fragment_source: Option<ShaderSource>,
    compute_source: Option<ShaderSource>,
    last_update: Instant,
}
impl ShaderSet {
    pub fn simple(
        watcher: &mut ShaderDirectoryWatcher,
        vertex_source: ShaderSource,
        fragment_source: ShaderSource,
    ) -> Result<Self, anyhow::Error> {
        Ok(Self {
            inner: ShaderSetInner::simple(
                vertex_source.load(&watcher.directory)?,
                fragment_source.load(&watcher.directory)?,
            )?,
            vertex_source: Some(vertex_source),
            fragment_source: Some(fragment_source),
            compute_source: None,
            last_update: Instant::now(),
        })
    }
    pub fn compute_only(
        watcher: &mut ShaderDirectoryWatcher,
        compute_source: ShaderSource,
    ) -> Result<Self, anyhow::Error> {
        Ok(Self {
            inner: ShaderSetInner::compute_only(compute_source.load(&watcher.directory)?)?,
            vertex_source: None,
            fragment_source: None,
            compute_source: Some(compute_source),
            last_update: Instant::now(),
        })
    }

    /// Refreshes the shader if necessary. Returns whether a refresh happened.
    pub fn refresh(&mut self, directory_watcher: &mut ShaderDirectoryWatcher) -> bool {
        directory_watcher.detect_changes();

        if ![&self.vertex_source, &self.fragment_source, &self.compute_source]
            .into_iter()
            .flat_map(|s| s.iter())
            .map(|s| &s.filenames)
            .flatten()
            .flatten()
            .filter_map(|n| directory_watcher.last_modifications.get(n))
            .any(|&t| t > self.last_update)
        {
            return false;
        }

        let r: Result<(), anyhow::Error> = try {
            self.inner = match (&self.vertex_source, &self.fragment_source, &self.compute_source) {
                (Some(ref vs), Some(ref fs), None) => ShaderSetInner::simple(
                    vs.load(&directory_watcher.directory)?,
                    fs.load(&directory_watcher.directory)?,
                ),
                (None, None, Some(ref cs)) => {
                    ShaderSetInner::compute_only(cs.load(&directory_watcher.directory)?)
                }
                _ => unreachable!(),
            }?;
        };
        self.last_update = Instant::now();
        r.is_ok()
    }

    pub fn layout_descriptor(&self) -> wgpu::BindGroupLayoutDescriptor {
        wgpu::BindGroupLayoutDescriptor {
            entries: self.inner.layout_descriptor[..].into(),
            label: None,
        }
    }
    pub fn desc_names(&self) -> &[Option<String>] {
        &self.inner.desc_names[..]
    }
    pub fn input_attributes(&self) -> &[wgpu::VertexAttributeDescriptor] {
        &self.inner.input_attributes[..]
    }

    pub fn vertex(&self) -> &[u32] {
        self.inner.vertex.as_ref().unwrap()
    }
    pub fn fragment(&self) -> &[u32] {
        self.inner.fragment.as_ref().unwrap()
    }
    pub fn compute(&self) -> &[u32] {
        self.inner.compute.as_ref().unwrap()
    }
}
