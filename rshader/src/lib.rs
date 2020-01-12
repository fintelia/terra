pub mod dynamic_shaders;
pub mod static_shaders;

use spirq::ty::{DescriptorType, ImageArrangement};
use spirq::SpirvBinary;
use std::collections::{btree_map::Entry, BTreeMap};

#[cfg(feature = "dynamic_shaders")]
pub use dynamic_shaders::*;
#[cfg(not(feature = "dynamic_shaders"))]
pub use static_shaders::*;

pub struct ShaderSource {
    pub source: Option<String>,
    pub filenames: Option<Vec<String>>,
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
                $( tmp_vec.push($filename.to_string()); )*
                    tmp_vec
            }),
        }
    };
}

fn create_vertex_shader(source: &str) -> Result<Vec<u8>, failure::Error> {
    let mut glsl_compiler = shaderc::Compiler::new().unwrap();
    Ok(glsl_compiler
        .compile_into_spirv(source, shaderc::ShaderKind::Vertex, "[VERTEX]", "main", None)?
        .as_binary_u8()
        .to_vec())
}
fn create_fragment_shader(source: &str) -> Result<Vec<u8>, failure::Error> {
    let mut glsl_compiler = shaderc::Compiler::new().unwrap();
    Ok(glsl_compiler
        .compile_into_spirv(source, shaderc::ShaderKind::Fragment, "[FRAGMENT]", "main", None)?
        .as_binary_u8()
        .to_vec())
}
fn create_compute_shader(source: &str) -> Result<Vec<u8>, failure::Error> {
    let mut glsl_compiler = shaderc::Compiler::new().unwrap();
    Ok(glsl_compiler
        .compile_into_spirv(source, shaderc::ShaderKind::Compute, "[COMPUTE]", "main", None)?
        .as_binary_u8()
        .to_vec())
}

fn reflect(
    stages: &[(wgpu::ShaderStage, &[u8])],
) -> Result<(Vec<Option<String>>, Vec<wgpu::BindGroupLayoutBinding>), failure::Error> {
    let mut binding_map: BTreeMap<u32, (Option<String>, wgpu::BindingType, wgpu::ShaderStage)> =
        BTreeMap::new();

    for (stage, spirv) in stages.iter() {
        let spv: SpirvBinary = spirv.to_vec().into();
        let entries = spv.reflect()?;
        let manifest = &entries[0].manifest;

        for desc in manifest.descs() {
            if let Some((set, binding)) = desc.desc_bind.into_inner() {
                assert_eq!(set, 0);
                let name = manifest.get_desc_name(desc.desc_bind).map(ToString::to_string);
                let ty = match desc.desc_ty {
                    DescriptorType::Sampler => wgpu::BindingType::Sampler,
                    DescriptorType::UniformBuffer(..) => {
                        wgpu::BindingType::UniformBuffer { dynamic: false }
                    }
                    DescriptorType::Image(spirq::ty::Type::Image(ty)) => {
                        wgpu::BindingType::SampledTexture {
                            multisampled: false,
                            dimension: match ty.arng {
                                ImageArrangement::Image2D => wgpu::TextureViewDimension::D2,
                                ImageArrangement::Image2DArray => {
                                    wgpu::TextureViewDimension::D2Array
                                }
                                _ => unimplemented!(),
                            },
                        }
                    }
                    v => unimplemented!("{:?}", v),
                };

                match binding_map.entry(binding) {
                    Entry::Vacant(v) => {
                        v.insert((name, ty, *stage));
                    }
                    Entry::Occupied(mut e) => {
                        let (ref n, ref t, ref mut s) = e.get_mut();
                        *s = *s | *stage;

                        if *n != name || *t != ty {
                            return Err(failure::format_err!("descriptor mismatch"));
                        }
                    }
                }
            }
        }
    }

    let mut names = Vec::new();
    let mut bindings = Vec::new();
    for (binding, (name, ty, visibility)) in binding_map.into_iter() {
        names.push(name);
        bindings.push(wgpu::BindGroupLayoutBinding { binding, visibility, ty });
    }

    Ok((names, bindings))
}
