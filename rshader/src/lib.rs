use anyhow::anyhow;
use naga::{
    AddressSpace, ImageClass, ImageDimension, ScalarKind, StorageAccess, StorageFormat, TypeInner,
};
use notify::{self, RecommendedWatcher, RecursiveMode, Watcher};
use std::collections::HashMap;
use std::collections::{btree_map::Entry, BTreeMap};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

pub enum ShaderSource {
    Inline {
        name: &'static str,
        contents: String,
        headers: HashMap<&'static str, String>,
        defines: Vec<(&'static str, &'static str)>,
    },
    Files {
        name: &'static str,
        path: PathBuf,
        header_paths: HashMap<&'static str, PathBuf>,
        defines: Vec<(&'static str, &'static str)>,
    },
    FilesWGSL {
        name: &'static str,
        path: PathBuf,
        header_paths: HashMap<&'static str, PathBuf>,
    },
}
impl ShaderSource {
    pub fn new(
        directory: PathBuf,
        name: &'static str,
        mut header_paths: HashMap<&'static str, PathBuf>,
        defines: Vec<(&'static str, &'static str)>,
    ) -> Self {
        DIRECTORY_WATCHER.lock().unwrap().watch(&directory);
        let path = std::fs::canonicalize(directory.join(&PathBuf::from(name))).unwrap();
        for header in header_paths.values_mut() {
            *header = std::fs::canonicalize(directory.join(&header)).unwrap();
        }
        ShaderSource::Files { name, path, header_paths, defines }
    }
    pub fn new_wgsl(
        directory: PathBuf,
        name: &'static str,
        mut header_paths: HashMap<&'static str, PathBuf>,
    ) -> Self {
        DIRECTORY_WATCHER.lock().unwrap().watch(&directory);
        let path = std::fs::canonicalize(directory.join(&PathBuf::from(name))).unwrap();
        for header in header_paths.values_mut() {
            *header = std::fs::canonicalize(directory.join(&header)).unwrap();
        }
        ShaderSource::FilesWGSL { name, path, header_paths }
    }
    pub(crate) fn load(
        &self,
        stage: naga::ShaderStage,
    ) -> Result<wgpu::ShaderSource<'static>, anyhow::Error> {
        let (name, contents, headers, defines) = match self {
            ShaderSource::Inline { name, contents, headers, defines } => {
                (name, contents.clone(), headers.clone(), Some(defines))
            }
            ShaderSource::Files { name, path, header_paths, defines } => {
                let file = std::fs::read_to_string(path)?;
                let mut headers = HashMap::new();
                for (&name, path) in header_paths.iter() {
                    headers.insert(name, std::fs::read_to_string(path)?);
                }
                (name, file, headers, Some(defines))
            }
            ShaderSource::FilesWGSL { name, path, header_paths } => {
                let mut file = String::new();
                for (_name, path) in header_paths.iter() {
                    file.push_str(&std::fs::read_to_string(path)?);
                }
                file.push_str(&std::fs::read_to_string(path)?);
                (name, file, HashMap::new(), None)
            }
        };

        if let ShaderSource::FilesWGSL { .. } = self {
            Ok(wgpu::ShaderSource::Wgsl(contents.into()))
        } else {
            let mut parser = naga::front::glsl::Parser::default();

            let mut combined_source = contents.clone();
            for (name, header_contents) in headers.iter() {
                combined_source = combined_source.replace(
                    &format!("\n#include \"{}\"", name),
                    &format!("\n{}", header_contents),
                );
            }

            let defines = defines
                .into_iter()
                .flatten()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            let module =
                parser.parse(&naga::front::glsl::Options { stage, defines }, &combined_source);

            match module {
                Err(e) => {
                    for e in e {
                        if let Some(range) = e.meta.to_range() {
                            eprintln!(
                                "ERROR: {:?} '{}'",
                                e.kind,
                                &combined_source[(range.start.max(30) - 30)..range.end]
                            );
                        } else {
                            eprintln!("ERROR: {:?}", e.kind);
                        }
                    }
                    Err(anyhow::anyhow!("Failed to parse shader"))
                }
                Ok(module) => {
                    let mut validator = naga::valid::Validator::new(
                        naga::valid::ValidationFlags::all(),
                        naga::valid::Capabilities::all(),
                    );
                    match validator.validate(&module) {
                        Err(e) => {
                            e.emit_to_stderr_with_path(&combined_source, name);
                            Err(anyhow::anyhow!("Failed to validate shader"))
                        }
                        Ok(_) => Ok(wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(module))),
                    }
                }
            }
        }
    }
    pub(crate) fn needs_update(&self, last_update: Instant) -> bool {
        match self {
            ShaderSource::Inline { .. } => false,
            ShaderSource::Files { path, header_paths, .. }
            | ShaderSource::FilesWGSL { path, header_paths, .. } => {
                let directory_watcher = DIRECTORY_WATCHER.lock().unwrap();
                header_paths
                    .values()
                    .chain(std::iter::once(path))
                    .filter_map(|f| directory_watcher.last_modifications.get(f))
                    .any(|&t| t > last_update)
            }
        }
    }
}

pub(crate) struct ShaderSetInner {
    pub vertex: Option<wgpu::ShaderSource<'static>>,
    pub fragment: Option<wgpu::ShaderSource<'static>>,
    pub compute: Option<wgpu::ShaderSource<'static>>,

    pub input_attributes: Vec<wgpu::VertexAttribute>,
    pub layout_descriptor: Vec<wgpu::BindGroupLayoutEntry>,
    pub desc_names: Vec<Option<String>>,
    pub workgroup_size: Option<[u32; 3]>,
}
impl ShaderSetInner {
    pub fn simple(
        vertex: wgpu::ShaderSource<'static>,
        fragment: wgpu::ShaderSource<'static>,
    ) -> Result<Self, anyhow::Error> {
        let (input_attributes, desc_names, layout_descriptor, _) =
            reflect_naga(&[&vertex, &fragment])?;

        Ok(Self {
            vertex: Some(vertex),
            fragment: Some(fragment),
            compute: None,
            desc_names,
            layout_descriptor,
            input_attributes,
            workgroup_size: None,
        })
    }

    pub fn compute_only(source: wgpu::ShaderSource<'static>) -> Result<Self, anyhow::Error> {
        let (input_attributes, desc_names, layout_descriptor, workgroup_size) =
            reflect_naga(&[&source])?;

        assert!(input_attributes.is_empty());
        Ok(Self {
            vertex: None,
            fragment: None,
            compute: Some(source),
            desc_names,
            layout_descriptor,
            input_attributes,
            workgroup_size: Some(workgroup_size),
        })
    }
}

pub(crate) struct DirectoryWatcher {
    watcher: RecommendedWatcher,
    last_modifications: HashMap<PathBuf, Instant>,
}
impl DirectoryWatcher {
    pub fn new() -> Self {
        let watcher = notify::recommended_watcher(|event: Result<notify::Event, _>| {
            if let Ok(event) = event {
                if event.kind.is_modify() {
                    let last_modifications =
                        &mut DIRECTORY_WATCHER.lock().unwrap().last_modifications;
                    for p in event.paths {
                        last_modifications.insert(p, Instant::now());
                    }
                }
            }
        })
        .unwrap();

        Self { watcher, last_modifications: HashMap::new() }
    }

    pub fn watch(&mut self, directory: &Path) {
        self.watcher
            .watch(directory, RecursiveMode::Recursive)
            .expect(&format!("rshader: Failed to watch path '{}'", directory.display()))
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
        vertex_source: ShaderSource,
        fragment_source: ShaderSource,
    ) -> Result<Self, anyhow::Error> {
        Ok(Self {
            inner: ShaderSetInner::simple(
                vertex_source.load(naga::ShaderStage::Vertex)?,
                fragment_source.load(naga::ShaderStage::Fragment)?,
            )?,
            vertex_source: Some(vertex_source),
            fragment_source: Some(fragment_source),
            compute_source: None,
            last_update: Instant::now(),
        })
    }
    pub fn compute_only(compute_source: ShaderSource) -> Result<Self, anyhow::Error> {
        Ok(Self {
            inner: ShaderSetInner::compute_only(compute_source.load(naga::ShaderStage::Compute)?)?,
            vertex_source: None,
            fragment_source: None,
            compute_source: Some(compute_source),
            last_update: Instant::now(),
        })
    }

    /// Refreshes the shader if necessary. Returns whether a refresh happened.
    pub fn refresh(&mut self) -> bool {
        if !self.vertex_source.as_ref().map(|s| s.needs_update(self.last_update)).unwrap_or(false)
            && !self
                .fragment_source
                .as_ref()
                .map(|s| s.needs_update(self.last_update))
                .unwrap_or(false)
            && !self
                .compute_source
                .as_ref()
                .map(|s| s.needs_update(self.last_update))
                .unwrap_or(false)
        {
            return false;
        }

        let r =
            || -> Result<(), anyhow::Error> {
                Ok(self.inner =
                    match (&self.vertex_source, &self.fragment_source, &self.compute_source) {
                        (Some(ref vs), Some(ref fs), None) => ShaderSetInner::simple(
                            vs.load(naga::ShaderStage::Vertex)?,
                            fs.load(naga::ShaderStage::Fragment)?,
                        ),
                        (None, None, Some(ref cs)) => {
                            ShaderSetInner::compute_only(cs.load(naga::ShaderStage::Compute)?)
                        }
                        _ => unreachable!(),
                    }?)
            }();
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
    pub fn input_attributes(&self) -> &[wgpu::VertexAttribute] {
        &self.inner.input_attributes[..]
    }

    pub fn vertex(&self) -> wgpu::ShaderSource {
        match self.inner.vertex.as_ref().unwrap().clone() {
            wgpu::ShaderSource::SpirV(s) => wgpu::ShaderSource::SpirV(s.clone()),
            wgpu::ShaderSource::Wgsl(w) => wgpu::ShaderSource::Wgsl(w.clone()),
            wgpu::ShaderSource::Naga(w) => wgpu::ShaderSource::Naga(w.clone()),
            _ => unreachable!(),
        }
    }
    pub fn fragment(&self) -> wgpu::ShaderSource {
        match self.inner.fragment.as_ref().unwrap() {
            wgpu::ShaderSource::SpirV(s) => wgpu::ShaderSource::SpirV(s.clone()),
            wgpu::ShaderSource::Wgsl(w) => wgpu::ShaderSource::Wgsl(w.clone()),
            wgpu::ShaderSource::Naga(w) => wgpu::ShaderSource::Naga(w.clone()),
            _ => unreachable!(),
        }
    }
    pub fn compute(&self) -> wgpu::ShaderSource {
        match self.inner.compute.as_ref().unwrap() {
            wgpu::ShaderSource::SpirV(s) => wgpu::ShaderSource::SpirV(s.clone()),
            wgpu::ShaderSource::Wgsl(w) => wgpu::ShaderSource::Wgsl(w.clone()),
            wgpu::ShaderSource::Naga(w) => wgpu::ShaderSource::Naga(w.clone()),
            _ => unreachable!(),
        }
    }
    pub fn workgroup_size(&self) -> [u32; 3] {
        self.inner.workgroup_size.unwrap()
    }
}

lazy_static::lazy_static! {
    static ref DIRECTORY_WATCHER: Mutex<DirectoryWatcher> = Mutex::new(DirectoryWatcher::new());
}

#[macro_export]
#[cfg(not(feature = "dynamic_shaders"))]
macro_rules! shader_source {
    ($directory:literal, $filename:literal $(, $header:literal )* $(; $define:literal = $value:literal )? ) => {
        {
            let contents = include_str!(concat!($directory, "/", $filename)).to_string();
            let mut headers = std::collections::HashMap::new();
            $( headers.insert($header, include_str!(concat!($directory, "/", $header)).to_string()); )*
            let mut defines = Vec::new();
            $( defines.push(($define, $value)); )*

            $crate::ShaderSource::Inline {
                name: $filename,
                contents,
                headers,
                defines,
            }
        }
    };
}

#[macro_export]
#[cfg(feature = "dynamic_shaders")]
macro_rules! shader_source {
    ($directory:literal, $filename:literal $(, $header:literal )* $(; $define:literal = $value:literal )? ) => {
		{
			let directory = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
				.join(file!()).parent().unwrap().join($directory);
			let mut headers = std::collections::HashMap::new();
			$( headers.insert($header, std::path::PathBuf::from($header)); )*
            let mut defines = Vec::new();
            $( defines.push(($define, $value)); )*

            $crate::ShaderSource::new(directory, $filename, headers, defines)
		}
    };
}

#[macro_export]
macro_rules! wgsl_source {
    ($directory:literal, $filename:literal $(, $header:literal )* ) => {
		{
			let directory = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
				.join(file!()).parent().unwrap().join($directory);
			let mut headers = std::collections::HashMap::new();
			$( headers.insert($header, std::path::PathBuf::from($header)); )*

            $crate::ShaderSource::new_wgsl(directory, $filename, headers)
		}
    };
}

fn reflect_naga(
    stages: &[&wgpu::ShaderSource<'static>],
) -> Result<
    (Vec<wgpu::VertexAttribute>, Vec<Option<String>>, Vec<wgpu::BindGroupLayoutEntry>, [u32; 3]),
    anyhow::Error,
> {
    let mut binding_map: BTreeMap<u32, (Option<String>, wgpu::BindingType, wgpu::ShaderStages)> =
        BTreeMap::new();

    // let mut attribute_offset = 0;
    // let mut attributes = Vec::new();
    let mut workgroup_size = None;
    for stage in stages.iter() {
        let module = match stage {
            wgpu::ShaderSource::SpirV(s) => naga::front::spv::parse_u8_slice(
                bytemuck::cast_slice(s),
                &naga::front::spv::Options {
                    adjust_coordinate_space: false,
                    strict_capabilities: false,
                    block_ctx_dump_prefix: None,
                },
            )?,
            wgpu::ShaderSource::Wgsl(w) => naga::front::wgsl::parse_str(w)?,
            wgpu::ShaderSource::Naga(ref m) => m.as_ref().clone(),
            _ => unreachable!(),
        };

        let _module_info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::FLOAT64 | naga::valid::Capabilities::PUSH_CONSTANT, /*naga::valid::Capabilities::empty()*/
        )
        .validate(&module)?;

        let stage = match module.entry_points[0].stage {
            naga::ShaderStage::Vertex => wgpu::ShaderStages::VERTEX,
            naga::ShaderStage::Fragment => wgpu::ShaderStages::FRAGMENT,
            naga::ShaderStage::Compute => wgpu::ShaderStages::COMPUTE,
        };

        workgroup_size = Some(module.entry_points[0].workgroup_size);

        // TODO: handle vertex attributes

        for (_handle, variable) in module.global_variables.iter() {
            let (_set, binding) = match &variable.binding {
                Some(r) => (r.group, r.binding),
                None => continue,
            };
            let mut name = variable.name.clone();
            let ty = &module.types.get_handle(variable.ty).unwrap().inner;

            // If this is an unnamed interface block, but it contains only a single named item,
            // use the item's name instead.
            if name.is_none() || name.as_ref().unwrap().is_empty() {
                if let TypeInner::Struct { members, .. } = ty {
                    if members.len() == 1 {
                        name = members[0].name.clone();
                    }
                }
            }

            let ty = match ty {
                TypeInner::Sampler { comparison } => wgpu::BindingType::Sampler(if *comparison {
                    wgpu::SamplerBindingType::Comparison
                } else {
                    wgpu::SamplerBindingType::Filtering
                }),
                TypeInner::Image { dim, arrayed, class } => {
                    let view_dimension = match (dim, arrayed) {
                        (ImageDimension::D1, false) => wgpu::TextureViewDimension::D1,
                        (ImageDimension::D2, false) => wgpu::TextureViewDimension::D2,
                        (ImageDimension::D3, false) => wgpu::TextureViewDimension::D3,
                        (ImageDimension::D2, true) => wgpu::TextureViewDimension::D2Array,
                        _ => unreachable!(),
                    };
                    match class {
                        ImageClass::Storage { format, access } => {
                            wgpu::BindingType::StorageTexture {
                                view_dimension,
                                access: if access.contains(StorageAccess::STORE)
                                    && access.contains(StorageAccess::LOAD)
                                {
                                    wgpu::StorageTextureAccess::ReadWrite
                                } else if access.contains(StorageAccess::STORE) {
                                    wgpu::StorageTextureAccess::WriteOnly
                                } else {
                                    wgpu::StorageTextureAccess::ReadOnly
                                },
                                format: match format {
                                    StorageFormat::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
                                    StorageFormat::R32Float => wgpu::TextureFormat::R32Float,
                                    StorageFormat::Rg32Float => wgpu::TextureFormat::Rg32Float,
                                    StorageFormat::Rgba32Float => wgpu::TextureFormat::Rgba32Float,
                                    StorageFormat::R32Uint => wgpu::TextureFormat::R32Uint,
                                    StorageFormat::Rg32Uint => wgpu::TextureFormat::Rg32Uint,
                                    StorageFormat::Rgba32Uint => wgpu::TextureFormat::Rgba32Uint,
                                    StorageFormat::R8Unorm => wgpu::TextureFormat::R8Unorm,
                                    StorageFormat::Rg8Unorm => wgpu::TextureFormat::Rg8Unorm,
                                    StorageFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
                                    _ => unimplemented!("format {:?}", format),
                                },
                            }
                        }
                        ImageClass::Sampled { kind, multi } => wgpu::BindingType::Texture {
                            multisampled: *multi,
                            view_dimension,
                            sample_type: match kind {
                                ScalarKind::Float => {
                                    wgpu::TextureSampleType::Float { filterable: true }
                                }
                                ScalarKind::Uint => wgpu::TextureSampleType::Uint,
                                ScalarKind::Sint => wgpu::TextureSampleType::Sint,
                                ScalarKind::Bool => unreachable!(),
                            },
                        },
                        ImageClass::Depth { multi } => wgpu::BindingType::Texture {
                            multisampled: *multi,
                            view_dimension,
                            sample_type: wgpu::TextureSampleType::Depth,
                        },
                    }
                }
                _ => match variable.space {
                    AddressSpace::Storage { access } => wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: !access.contains(StorageAccess::STORE),
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    AddressSpace::Uniform => wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    _ => continue,
                },
            };

            match binding_map.entry(binding) {
                Entry::Vacant(v) => {
                    v.insert((name, ty, stage));
                }
                Entry::Occupied(mut e) => {
                    let (ref n, ref t, ref mut s) = e.get_mut();
                    *s = *s | stage;

                    if *n != name {
                        return Err(anyhow!(
                            "descriptor mismatch {} vs {}",
                            n.as_ref().unwrap_or(&"<unamed>".to_string()),
                            name.unwrap_or("<unamed>".to_string())
                        ));
                    }
                    if *t != ty {
                        return Err(anyhow!(
                            "descriptor mismatch for {}: {:?} vs {:?}",
                            n.as_ref().unwrap_or(&"<unamed>".to_string()),
                            t,
                            ty
                        ));
                    }
                }
            }
        }
    }

    let mut names = Vec::new();
    let mut bindings = Vec::new();
    for (binding, (name, ty, visibility)) in binding_map.into_iter() {
        names.push(name);
        bindings.push(wgpu::BindGroupLayoutEntry { binding, visibility, ty, count: None });
    }

    Ok((Vec::new(), names, bindings, workgroup_size.unwrap()))
}
