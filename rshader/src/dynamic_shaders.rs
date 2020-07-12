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
    vertex: Option<Vec<u32>>,
    fragment: Option<Vec<u32>>,
    compute: Option<Vec<u32>>,

    input_attributes: Vec<wgpu::VertexAttributeDescriptor>,
    layout_descriptor: Vec<wgpu::BindGroupLayoutEntry>,
    desc_names: Vec<Option<String>>,

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

        let (input_attributes, desc_names, layout_descriptor) = crate::reflect(&[&vertex[..], &fragment[..]])?;

        Ok(Self {
            vertex: Some(vertex),
            fragment: Some(fragment),
            compute: None,
            last_update: Instant::now(),
            vertex_filenames: vertex_filenames,
            fragment_filenames: fragment_filenames,
            compute_filenames: Vec::new(),
            desc_names,
            layout_descriptor,
            input_attributes,
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

        let (input_attributes, desc_names, layout_descriptor) = crate::reflect(&[&compute[..]])?;

        assert!(input_attributes.is_empty());

        Ok(Self {
            vertex: None,
            fragment: None,
            compute: Some(compute),
            last_update: Instant::now(),
            vertex_filenames: Vec::new(),
            fragment_filenames: Vec::new(),
            compute_filenames: compute_filenames,
            desc_names,
            layout_descriptor,
            input_attributes,
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
                let mut stages = Vec::new();
                if !self.vertex_filenames.is_empty() {
                    vs = Some(create_vertex_shader(&concat_file_contents(
                        self.vertex_filenames.iter(),
                    )?)?);
                    stages.push(&vs.as_ref().unwrap()[..]);
                }
                if !self.fragment_filenames.is_empty() {
                    fs = Some(create_fragment_shader(&concat_file_contents(
                        self.fragment_filenames.iter(),
                    )?)?);
                    stages.push(&fs.as_ref().unwrap()[..]);
                }
                if !self.compute_filenames.is_empty() {
                    cs = Some(create_compute_shader(&concat_file_contents(
                        self.compute_filenames.iter(),
                    )?)?);
                    stages.push(&cs.as_ref().unwrap()[..]);
                }

                let (ia, dn, ld) = crate::reflect(&stages[..])?;

                Ok((vs, fs, cs, ia, dn, ld))
            }();

            if let Ok((vs, fs, cs, ia, dn, ld)) = new_shaders {
                self.input_attributes = ia;
                self.desc_names = dn;
                self.layout_descriptor = ld;
                self.vertex = vs;
                self.fragment = fs;
                self.compute = cs;
                return true;
            }
        }
        false
    }

    pub fn layout_descriptor(&self) -> wgpu::BindGroupLayoutDescriptor {
        wgpu::BindGroupLayoutDescriptor { bindings: &self.layout_descriptor[..], label: None }
    }
    pub fn desc_names(&self) -> &[Option<String>] {
        &self.desc_names[..]
    }
    pub fn input_attributes(&self) -> &[wgpu::VertexAttributeDescriptor] {
        &self.input_attributes[..]
    }

    pub fn vertex(&self) -> &[u32] {
        self.vertex.as_ref().unwrap()
    }
    pub fn fragment(&self) -> &[u32] {
        self.fragment.as_ref().unwrap()
    }
    pub fn compute(&self) -> &[u32] {
        self.compute.as_ref().unwrap()
    }
}
