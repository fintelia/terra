use anyhow::anyhow;
use notify::{self, DebouncedEvent, RecommendedWatcher, RecursiveMode, Watcher};
use spirq::ty::{DescriptorType, ImageArrangement, ScalarType, Type, VectorType};
use spirq::{ExecutionModel, SpirvBinary};
use spirv_headers::ImageFormat;
use std::collections::HashMap;
use std::collections::{btree_map::Entry, BTreeMap};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::sync::Mutex;
use std::time::{Duration, Instant};
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
    pub(crate) fn load(
        &self,
    ) -> Result<
        (&str, String, HashMap<&'static str, String>, &[(&'static str, &'static str)]),
        anyhow::Error,
    > {
        match self {
            ShaderSource::Inline { name, contents, headers, defines } => {
                Ok((&name, contents.clone(), headers.clone(), defines))
            }
            ShaderSource::Files { name, path, header_paths, defines } => {
                let file = std::fs::read_to_string(path)?;
                let mut headers = HashMap::new();
                for (&name, path) in header_paths.iter() {
                    headers.insert(name, std::fs::read_to_string(path)?);
                }
                Ok((&name, file, headers, defines))
            }
        }
    }
    pub(crate) fn needs_update(&self, last_update: Instant) -> bool {
        match self {
            ShaderSource::Inline { .. } => false,
            ShaderSource::Files { path, header_paths, .. } => {
                let mut directory_watcher = DIRECTORY_WATCHER.lock().unwrap();
                directory_watcher.detect_changes();
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
    pub vertex: Option<Vec<u32>>,
    pub fragment: Option<Vec<u32>>,
    pub compute: Option<Vec<u32>>,

    pub input_attributes: Vec<wgpu::VertexAttribute>,
    pub layout_descriptor: Vec<wgpu::BindGroupLayoutEntry>,
    pub desc_names: Vec<Option<String>>,
}
impl ShaderSetInner {
    pub fn simple(
        vsrc: (&str, String, HashMap<&'static str, String>, &[(&'static str, &'static str)]),
        fsrc: (&str, String, HashMap<&'static str, String>, &[(&'static str, &'static str)]),
    ) -> Result<Self, anyhow::Error> {
        let vertex = create_shader(vsrc.0, &vsrc.1, vsrc.2, vsrc.3, shaderc::ShaderKind::Vertex)?;
        let fragment =
            create_shader(fsrc.0, &fsrc.1, fsrc.2, vsrc.3, shaderc::ShaderKind::Fragment)?;
        let (input_attributes, desc_names, layout_descriptor) =
            crate::reflect(&[&vertex[..], &fragment[..]])?;

        Ok(Self {
            vertex: Some(vertex),
            fragment: Some(fragment),
            compute: None,
            desc_names,
            layout_descriptor,
            input_attributes,
        })
    }

    pub fn compute_only(
        src: (&str, String, HashMap<&'static str, String>, &[(&'static str, &'static str)]),
    ) -> Result<Self, anyhow::Error> {
        let compute = create_shader(src.0, &src.1, src.2, src.3, shaderc::ShaderKind::Compute)?;
        let (input_attributes, desc_names, layout_descriptor) = crate::reflect(&[&compute[..]])?;
        assert!(input_attributes.is_empty());

        Ok(Self {
            vertex: None,
            fragment: None,
            compute: Some(compute),
            desc_names,
            layout_descriptor,
            input_attributes,
        })
    }
}

pub(crate) struct DirectoryWatcher {
    watcher: RecommendedWatcher,
    watcher_rx: Receiver<DebouncedEvent>,
    last_modifications: HashMap<PathBuf, Instant>,
}
impl DirectoryWatcher {
    pub fn new() -> Self {
        let (tx, watcher_rx) = mpsc::channel();
        let watcher = notify::watcher(tx, Duration::from_millis(50)).unwrap();

        Self { watcher, watcher_rx, last_modifications: HashMap::new() }
    }

    pub fn detect_changes(&mut self) {
        while let Ok(event) = self.watcher_rx.try_recv() {
            if let DebouncedEvent::Write(p) = event {
                self.last_modifications.insert(p, Instant::now());
            }
        }
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
            inner: ShaderSetInner::simple(vertex_source.load()?, fragment_source.load()?)?,
            vertex_source: Some(vertex_source),
            fragment_source: Some(fragment_source),
            compute_source: None,
            last_update: Instant::now(),
        })
    }
    pub fn compute_only(compute_source: ShaderSource) -> Result<Self, anyhow::Error> {
        Ok(Self {
            inner: ShaderSetInner::compute_only(compute_source.load()?)?,
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
                        (Some(ref vs), Some(ref fs), None) => {
                            ShaderSetInner::simple(vs.load()?, fs.load()?)
                        }
                        (None, None, Some(ref cs)) => ShaderSetInner::compute_only(cs.load()?),
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

fn create_shader(
    input_file_name: &str,
    source_text: &str,
    headers: HashMap<&'static str, String>,
    defines: &[(&'static str, &'static str)],
    stage: shaderc::ShaderKind,
) -> Result<Vec<u32>, anyhow::Error> {
    let mut glsl_compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.set_include_callback(|f, _, _, _| match headers.get(f) {
        Some(s) => {
            Ok(shaderc::ResolvedInclude { resolved_name: f.to_string(), content: s.clone() })
        }
        None => Err("not found".to_string()),
    });
    for (m, value) in defines {
        options.add_macro_definition(m, Some(value));
    }

    Ok(glsl_compiler
        .compile_into_spirv(source_text, stage, input_file_name, "main", Some(&options))?
        .as_binary()
        .to_vec())
}

fn reflect(
    stages: &[&[u32]],
) -> Result<
    (Vec<wgpu::VertexAttribute>, Vec<Option<String>>, Vec<wgpu::BindGroupLayoutEntry>),
    anyhow::Error,
> {
    let mut binding_map: BTreeMap<u32, (Option<String>, wgpu::BindingType, wgpu::ShaderStages)> =
        BTreeMap::new();

    let mut attribute_offset = 0;
    let mut attributes = Vec::new();
    for spirv in stages.iter() {
        // let module = naga::front::spv::parse_u8_slice(bytemuck::cast_slice(spirv), &naga::front::spv::Options{
        //     adjust_coordinate_space: false,
        //     strict_capabilities: false,
        //     flow_graph_dump_prefix: None,
        // }).expect("Failed to parse with naga");

        let spv: SpirvBinary = spirv.to_vec().into();
        let entries = spv.reflect_vec()?;
        let manifest = &entries[0].manifest;

        let stage = match entries[0].exec_model {
            ExecutionModel::Vertex => wgpu::ShaderStages::VERTEX,
            ExecutionModel::Fragment => wgpu::ShaderStages::FRAGMENT,
            ExecutionModel::GLCompute => wgpu::ShaderStages::COMPUTE,
            _ => unimplemented!(),
        };

        if let wgpu::ShaderStages::VERTEX = stage {
            let mut inputs = BTreeMap::new();
            for input in manifest.inputs() {
                inputs.entry(u32::from(input.location.loc())).or_insert(Vec::new()).push(input);
            }
            for (shader_location, mut input) in inputs {
                input.sort_by_key(|i| u32::from(i.location.comp()));
                let i = input.last().unwrap();
                let (scalar_ty, nscalar) = match i.ty {
                    Type::Scalar(s) => (s, 1),
                    Type::Vector(VectorType { scalar_ty, nscalar }) => (scalar_ty, *nscalar),
                    _ => return Err(anyhow!("Unsupported attribute type")),
                };
                let (format, nbytes) = match (scalar_ty, nscalar + u32::from(i.location.comp())) {
                    (ScalarType::Signed(4), 1) => (wgpu::VertexFormat::Sint32, 4),
                    (ScalarType::Signed(4), 2) => (wgpu::VertexFormat::Sint32x2, 8),
                    (ScalarType::Signed(4), 3) => (wgpu::VertexFormat::Sint32x3, 12),
                    (ScalarType::Signed(4), 4) => (wgpu::VertexFormat::Sint32x4, 16),
                    (ScalarType::Unsigned(4), 1) => (wgpu::VertexFormat::Uint32, 4),
                    (ScalarType::Unsigned(4), 2) => (wgpu::VertexFormat::Uint32x2, 8),
                    (ScalarType::Unsigned(4), 3) => (wgpu::VertexFormat::Uint32x3, 12),
                    (ScalarType::Unsigned(4), 4) => (wgpu::VertexFormat::Uint32x4, 16),
                    (ScalarType::Float(4), 1) => (wgpu::VertexFormat::Float32, 4),
                    (ScalarType::Float(4), 2) => (wgpu::VertexFormat::Float32x2, 8),
                    (ScalarType::Float(4), 3) => (wgpu::VertexFormat::Float32x3, 12),
                    (ScalarType::Float(4), 4) => (wgpu::VertexFormat::Float32x4, 16),
                    _ => return Err(anyhow!("Unsupported attribute type")),
                };

                attributes.push(wgpu::VertexAttribute {
                    offset: attribute_offset,
                    format,
                    shader_location,
                });
                attribute_offset += nbytes;
            }
        }

        for desc in manifest.descs() {
            let (set, binding) = desc.desc_bind.into_inner();
            assert_eq!(set, 0);
            let mut name = manifest.get_desc_name(desc.desc_bind).map(ToString::to_string);

            // If this is an unnamed interface block, but it contains only a single named item,
            // use the item's name instead.
            if name.is_none() {
                match desc.desc_ty {
                    DescriptorType::UniformBuffer(_, Type::Struct(s))
                    | DescriptorType::StorageBuffer(_, Type::Struct(s)) => {
                        if s.nmember() == 1 {
                            name = s.get_member(0).unwrap().name.clone();
                        }
                    }
                    _ => {}
                }
            }

            let ty = match desc.desc_ty {
                DescriptorType::Sampler(_) => {
                    wgpu::BindingType::Sampler { filtering: true, comparison: false }
                }
                DescriptorType::UniformBuffer(..) => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                DescriptorType::Image(_, spirq::ty::Type::Image(ty)) => {
                    let view_dimension = match ty.arng {
                        ImageArrangement::Image2D => wgpu::TextureViewDimension::D2,
                        ImageArrangement::Image2DArray => wgpu::TextureViewDimension::D2Array,
                        ImageArrangement::Image3D => wgpu::TextureViewDimension::D3,
                        _ => unimplemented!(),
                    };
                    match ty.unit_fmt {
                        spirq::ty::ImageUnitFormat::Color(c) => wgpu::BindingType::StorageTexture {
                            view_dimension,
                            access: match manifest.get_desc_access(desc.desc_bind).unwrap() {
                                spirq::AccessType::ReadOnly => wgpu::StorageTextureAccess::ReadOnly,
                                spirq::AccessType::WriteOnly => {
                                    wgpu::StorageTextureAccess::WriteOnly
                                }
                                spirq::AccessType::ReadWrite => {
                                    wgpu::StorageTextureAccess::ReadWrite
                                }
                            },
                            format: match c {
                                ImageFormat::Rgba16f => wgpu::TextureFormat::Rgba16Float,
                                ImageFormat::R32f => wgpu::TextureFormat::R32Float,
                                ImageFormat::Rg32f => wgpu::TextureFormat::Rg32Float,
                                ImageFormat::Rgba32f => wgpu::TextureFormat::Rgba32Float,
                                ImageFormat::R32ui => wgpu::TextureFormat::R32Uint,
                                ImageFormat::Rg32ui => wgpu::TextureFormat::Rg32Uint,
                                ImageFormat::Rgba32ui => wgpu::TextureFormat::Rgba32Uint,
                                ImageFormat::Rgba8 => wgpu::TextureFormat::Rgba8Unorm,
                                _ => unimplemented!("component type {:?}", c),
                            },
                        },
                        spirq::ty::ImageUnitFormat::Sampled => wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        spirq::ty::ImageUnitFormat::Depth => unimplemented!(),
                    }
                }
                DescriptorType::StorageBuffer(..) => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: manifest.get_desc_access(desc.desc_bind).unwrap()
                            == spirq::AccessType::ReadOnly || stages.len() > 1, // TODO: Remove hack of assuming len=2 -> vertex+fragment -> readonly buffers
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                v => unimplemented!("{:?}", v),
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

    Ok((attributes, names, bindings))
}
