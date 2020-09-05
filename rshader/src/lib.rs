#![feature(try_blocks)]

pub mod dynamic_shaders;
pub mod static_shaders;

use anyhow::anyhow;
use sha2::{Digest, Sha256};
use spirq::ty::{DescriptorType, ImageArrangement, ScalarType, Type, VectorType};
use spirq::{ExecutionModel, SpirvBinary};
use std::collections::{btree_map::Entry, BTreeMap};
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};

#[cfg(feature = "dynamic_shaders")]
pub use dynamic_shaders::*;
#[cfg(not(feature = "dynamic_shaders"))]
pub use static_shaders::*;

pub struct ShaderSource {
    pub source: Option<String>,
    pub filenames: Option<Vec<PathBuf>>,
}
impl ShaderSource {
    pub fn load(&self, base_directory: &Path) -> Result<String, anyhow::Error> {
        if let Some(ref src) = self.source {
            return Ok(src.clone());
        }

        let mut contents = String::new();
        for filename in self.filenames.as_ref().unwrap() {
            File::open(fs::canonicalize(base_directory.join(filename))?)?
                .read_to_string(&mut contents)?;
        }
        Ok(contents)
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

#[macro_export]
#[cfg(not(feature = "dynamic_shaders"))]
macro_rules! shader_source {
    ($directory:expr, $( $filename:expr ),+ ) => {
        $crate::ShaderSource{
            source: Some({
                let mut tmp = String::new();
                $( tmp.push_str(
                    include_str!(concat!($directory, "/", $filename)));
                )*
                tmp
            }),
            filenames: None,
        }
    };
}

#[macro_export]
#[cfg(feature = "dynamic_shaders")]
macro_rules! shader_source {
    ($directory:expr, $( $filename:expr ),+ ) => {
        $crate::ShaderSource {
            source: None,
            filenames: Some({
                let mut tmp_vec = Vec::new();
                $( tmp_vec.push(std::path::Path::from($filename)); )*
                    tmp_vec
            }),
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
                input.sort_by_key(|i| u32::from(i.location.bind()));
                let i = input.last().unwrap();
                let (scalar_ty, nscalar) = match i.ty {
                    Type::Scalar(s) => (s, 1),
                    Type::Vector(VectorType { scalar_ty, nscalar }) => (scalar_ty, *nscalar),
                    _ => return Err(anyhow!("Unsupported attribute type")),
                };
                let (format, nbytes) = match (scalar_ty, nscalar + u32::from(i.location.bind())) {
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
                DescriptorType::Sampler(_) => wgpu::BindingType::Sampler { comparison: false },
                DescriptorType::UniformBuffer(..) => {
                    wgpu::BindingType::UniformBuffer { dynamic: false, min_binding_size: None }
                }
                DescriptorType::Image(_, spirq::ty::Type::Image(ty)) => {
                    wgpu::BindingType::SampledTexture {
                        multisampled: false,
                        dimension: match ty.arng {
                            ImageArrangement::Image2D => wgpu::TextureViewDimension::D2,
                            ImageArrangement::Image2DArray => wgpu::TextureViewDimension::D2Array,
                            ImageArrangement::Image3D => wgpu::TextureViewDimension::D3,
                            _ => unimplemented!(),
                        },
                        component_type: wgpu::TextureComponentType::Uint,
                    }
                }
                v => unimplemented!("{:?}", v),
            };

            match binding_map.entry(binding) {
                Entry::Vacant(v) => {
                    v.insert((name, ty, stage));
                }
                Entry::Occupied(mut e) => {
                    let (ref n, ref t, ref mut s) = e.get_mut();
                    *s = *s | stage;

                    if *n != name || *t != ty {
                        return Err(anyhow!("descriptor mismatch"));
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
