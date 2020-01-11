pub mod dynamic_shaders;
pub mod static_shaders;

use spirq::ty::{DescriptorType, ImageArrangement};

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
    spirv: &[u8],
) -> Result<(Vec<Option<String>>, Vec<wgpu::BindGroupLayoutBinding>), failure::Error> {
    let spv: spirq::SpirvBinary = spirv.to_vec().into();
    let entries = spv.reflect()?;
    let manifest = &entries[0].manifest;

    let mut names = Vec::new();
    let mut bindings = Vec::new();
    for desc in manifest.descs() {
        if let Some((set, binding)) = desc.desc_bind.into_inner() {
            assert_eq!(set, 0);

            names.push(manifest.get_desc_name(desc.desc_bind).map(ToString::to_string));
            bindings.push(wgpu::BindGroupLayoutBinding {
                binding,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: match desc.desc_ty {
                    DescriptorType::Sampler => wgpu::BindingType::Sampler,
                    DescriptorType::UniformBuffer(..) => {
                        wgpu::BindingType::UniformBuffer { dynamic: false }
                    }
                    DescriptorType::Image(spirq::ty::Type::Image(ty)) => wgpu::BindingType::SampledTexture {
                        multisampled: false,
                        dimension: match ty.arng {
                            ImageArrangement::Image2D => wgpu::TextureViewDimension::D2,
                            ImageArrangement::Image2DArray => {
                                wgpu::TextureViewDimension::D2Array
                            }
                            _ => unimplemented!(),
                        },
                    },
                    v => unimplemented!("{:?}", v),
                },
            });
        }
    }

    Ok((names, bindings))
}
