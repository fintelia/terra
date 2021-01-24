use anyhow::anyhow;
use notify::{self, DebouncedEvent, RecommendedWatcher, RecursiveMode, Watcher};
use sha2::{Digest, Sha256};
use spirq::ty::{DescriptorType, ImageArrangement, ScalarType, Type, VectorType};
use spirq::{ExecutionModel, SpirvBinary};
use std::collections::HashMap;
use std::collections::{btree_map::Entry, BTreeMap};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use spirv_headers::ImageFormat;
pub enum ShaderSource {
    Inline(String),
    Files(Vec<PathBuf>),
}
impl ShaderSource {
    pub fn new(directory: PathBuf, mut filenames: Vec<PathBuf>) -> Self {
        DIRECTORY_WATCHER.lock().unwrap().watch(&directory);

        for filename in &mut filenames {
            *filename = std::fs::canonicalize(directory.join(&filename)).unwrap();
        }
        ShaderSource::Files(filenames)
    }
    pub(crate) fn load(&self) -> Result<String, anyhow::Error> {
        match self {
            ShaderSource::Inline(s) => Ok(s.clone()),
            ShaderSource::Files(fs) => {
                let mut contents = String::new();
                for filename in fs {
                    File::open(filename)?.read_to_string(&mut contents)?;
                }
                Ok(contents)
            }
        }
    }
    pub(crate) fn needs_update(&self, last_update: Instant) -> bool {
        match self {
            ShaderSource::Inline(_) => false,
            ShaderSource::Files(v) => {
                let mut directory_watcher = DIRECTORY_WATCHER.lock().unwrap();
                directory_watcher.detect_changes();
                v.iter()
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

    pub input_attributes: Vec<wgpu::VertexAttributeDescriptor>,
    pub layout_descriptor: Vec<wgpu::BindGroupLayoutEntry>,
    pub desc_names: Vec<Option<String>>,
    pub digest: Vec<u8>,
}
impl ShaderSetInner {
    pub fn simple(vsrc: String, fsrc: String) -> Result<Self, anyhow::Error> {
        let vertex = create_vertex_shader(&vsrc)?;
        let fragment = create_fragment_shader(&fsrc)?;
        let digest = Sha256::digest(format!("v={}\0f={}", vsrc, fsrc).as_bytes()).to_vec();
        let (input_attributes, desc_names, layout_descriptor) =
            crate::reflect(&[&vertex[..], &fragment[..]])?;

        Ok(Self {
            vertex: Some(vertex),
            fragment: Some(fragment),
            compute: None,
            desc_names,
            layout_descriptor,
            input_attributes,
            digest,
        })
    }

    pub fn compute_only(src: String) -> Result<Self, anyhow::Error> {
        let compute = create_compute_shader(&src)?;
        let digest = Sha256::digest(format!("c={}", src).as_bytes()).to_vec();
        let (input_attributes, desc_names, layout_descriptor) = crate::reflect(&[&compute[..]])?;
        assert!(input_attributes.is_empty());

        Ok(Self {
            vertex: None,
            fragment: None,
            compute: Some(compute),
            desc_names,
            layout_descriptor,
            input_attributes,
            digest,
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

        let r = || -> Result<(), anyhow::Error> {
            Ok(self.inner = match (&self.vertex_source, &self.fragment_source, &self.compute_source) {
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
    pub fn digest(&self) -> &[u8] {
        &self.inner.digest
    }
}

lazy_static::lazy_static! {
    static ref DIRECTORY_WATCHER: Mutex<DirectoryWatcher> = Mutex::new(DirectoryWatcher::new());
}

#[macro_export]
#[cfg(not(feature = "dynamic_shaders"))]
macro_rules! shader_source {
    ($directory:expr, $( $filename:expr ),+ ) => {
        $crate::ShaderSource::Inline({
            let mut tmp = String::new();
            $( tmp.push_str(
                include_str!(concat!($directory, "/", $filename)));
            )*
                tmp
        })
    };
}

#[macro_export]
#[cfg(feature = "dynamic_shaders")]
macro_rules! shader_source {
    ($directory:expr, $( $filename:expr ),+ ) => {
		{
			let directory = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
				.join(file!()).parent().unwrap().join($directory);
			let mut files = Vec::new();
			$( files.push(std::path::PathBuf::from($filename)); )*

			$crate::ShaderSource::new(directory, files)
		}
    };
}

fn create_vertex_shader(source: &str) -> Result<Vec<u32>, anyhow::Error> {
    let mut glsl_compiler = shaderc::Compiler::new().unwrap();
    Ok(glsl_compiler
        .compile_into_spirv(source, shaderc::ShaderKind::Vertex, "[VERTEX]", "main", None)?
        .as_binary()
        .to_vec())
}
fn create_fragment_shader(source: &str) -> Result<Vec<u32>, anyhow::Error> {
    let mut glsl_compiler = shaderc::Compiler::new().unwrap();
    Ok(glsl_compiler
        .compile_into_spirv(source, shaderc::ShaderKind::Fragment, "[FRAGMENT]", "main", None)?
        .as_binary()
        .to_vec())
}
fn create_compute_shader(source: &str) -> Result<Vec<u32>, anyhow::Error> {
    let mut glsl_compiler = shaderc::Compiler::new().unwrap();
    Ok(glsl_compiler
        .compile_into_spirv(source, shaderc::ShaderKind::Compute, "[COMPUTE]", "main", None)?
        .as_binary()
        .to_vec())
}

fn reflect(
    stages: &[&[u32]],
) -> Result<
    (Vec<wgpu::VertexAttributeDescriptor>, Vec<Option<String>>, Vec<wgpu::BindGroupLayoutEntry>),
    anyhow::Error,
> {
    let mut binding_map: BTreeMap<u32, (Option<String>, wgpu::BindingType, wgpu::ShaderStage)> =
        BTreeMap::new();

    let mut attribute_offset = 0;
    let mut attributes = Vec::new();
    for spirv in stages.iter() {
        let spv: SpirvBinary = spirv.to_vec().into();
        let entries = spv.reflect()?;
        let manifest = &entries[0].manifest;

        let stage = match entries[0].exec_model {
            ExecutionModel::Vertex => wgpu::ShaderStage::VERTEX,
            ExecutionModel::Fragment => wgpu::ShaderStage::FRAGMENT,
            ExecutionModel::GLCompute => wgpu::ShaderStage::COMPUTE,
            _ => unimplemented!(),
        };

        if let wgpu::ShaderStage::VERTEX = stage {
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
                    (ScalarType::Signed(4), 1) => (wgpu::VertexFormat::Int, 4),
                    (ScalarType::Signed(4), 2) => (wgpu::VertexFormat::Int2, 8),
                    (ScalarType::Signed(4), 3) => (wgpu::VertexFormat::Int3, 12),
                    (ScalarType::Signed(4), 4) => (wgpu::VertexFormat::Int4, 16),
                    (ScalarType::Unsigned(4), 1) => (wgpu::VertexFormat::Uint, 4),
                    (ScalarType::Unsigned(4), 2) => (wgpu::VertexFormat::Uint2, 8),
                    (ScalarType::Unsigned(4), 3) => (wgpu::VertexFormat::Uint3, 12),
                    (ScalarType::Unsigned(4), 4) => (wgpu::VertexFormat::Uint4, 16),
                    (ScalarType::Float(4), 1) => (wgpu::VertexFormat::Float, 4),
                    (ScalarType::Float(4), 2) => (wgpu::VertexFormat::Float2, 8),
                    (ScalarType::Float(4), 3) => (wgpu::VertexFormat::Float3, 12),
                    (ScalarType::Float(4), 4) => (wgpu::VertexFormat::Float4, 16),
                    _ => return Err(anyhow!("Unsupported attribute type")),
                };

                attributes.push(wgpu::VertexAttributeDescriptor {
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
            let name = manifest.get_desc_name(desc.desc_bind).map(ToString::to_string);
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
                        spirq::ty::ImageUnitFormat::Color(c) => {
                            wgpu::BindingType::StorageTexture {
                                view_dimension,
                                access: match manifest.get_desc_access(desc.desc_bind).unwrap() {
                                    spirq::AccessType::ReadOnly => wgpu::StorageTextureAccess::ReadOnly,
                                    spirq::AccessType::WriteOnly => wgpu::StorageTextureAccess::WriteOnly,
                                    spirq::AccessType::ReadWrite => wgpu::StorageTextureAccess::ReadWrite,
                                },
                                format: match c {
                                    ImageFormat::R32f => wgpu::TextureFormat::R32Float,
                                    ImageFormat::Rg32f => wgpu::TextureFormat::Rg32Float,
                                    ImageFormat::Rgba32f => wgpu::TextureFormat::Rgba32Float,
                                    ImageFormat::R32ui => wgpu::TextureFormat::R32Uint,
                                    ImageFormat::Rg32ui => wgpu::TextureFormat::Rg32Uint,
                                    ImageFormat::Rgba32ui => wgpu::TextureFormat::Rgba32Uint,
                                    ImageFormat::Rgba8 => wgpu::TextureFormat::Rgba8Unorm,
                                    _ => unimplemented!("component type {:?}", c),
                                }
                            }
                        }
                        spirq::ty::ImageUnitFormat::Sampled => wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        spirq::ty::ImageUnitFormat::Depth => unimplemented!(),
                    }
                }
                DescriptorType::StorageBuffer(..) => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false, },
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
                        return Err(anyhow!("descriptor mismatch {} vs {}",
                            n.as_ref().unwrap_or(&"<unamed>".to_string()), name.unwrap_or("<unamed>".to_string())));
                    }
                    if *t != ty {
                        return Err(anyhow!("descriptor mismatch for {}: {:?} vs {:?}",
                            n.as_ref().unwrap_or(&"<unamed>".to_string()), t, ty));
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
