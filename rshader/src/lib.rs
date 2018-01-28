extern crate gfx;
extern crate failure;
extern crate notify;

pub mod static_shaders;
pub mod dynamic_shaders;

#[cfg(not(feature = "dynamic_shaders"))]
pub use static_shaders::*;
#[cfg(feature = "dynamic_shaders")]
pub use dynamic_shaders::*;

pub struct ShaderSource {
    pub source: Option<Vec<u8>>,
    pub filenames: Option<Vec<String>>,
}

#[macro_export]
#[cfg(not(feature = "dynamic_shaders"))]
macro_rules! shader_source {
    ($directory:expr, $( $filename:expr ),+ ) => {
        $crate::ShaderSource{
            source: Some({
                let mut tmp_vec = Vec::new();
                $( tmp_vec.extend_from_slice(
                    include_bytes!(concat!($directory, "/", $filename)));
                )*
                tmp_vec
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

fn print_shader_error(error: &gfx::shade::core::CreateShaderError) {
    use gfx::shade::core::CreateShaderError::*;
    match *error {
        ModelNotSupported => eprintln!("Attempted to use unsupported shader model"),
        StageNotSupported(ref stage) => eprintln!("Shader stage '{:?}' not supported", stage),
        CompilationFailed(ref msg) => eprintln!("Shader complilation failed: \n{}", msg),
    }
}

fn create_vertex_shader<R: gfx::Resources, F: gfx::Factory<R>>(
    factory: &mut F,
    source: &[u8],
) -> Result<gfx::VertexShader<R>, gfx::shade::core::CreateShaderError> {
    match factory.create_shader_vertex(source) {
        Ok(shader) => Ok(shader),
        Err(error) => {
            print_shader_error(&error);
            Err(error)
        }
    }
}
fn create_pixel_shader<R: gfx::Resources, F: gfx::Factory<R>>(
    factory: &mut F,
    source: &[u8],
) -> Result<gfx::PixelShader<R>, gfx::shade::core::CreateShaderError> {
    match factory.create_shader_pixel(source) {
        Ok(shader) => Ok(shader),
        Err(error) => {
            print_shader_error(&error);
            Err(error)
        }
    }
}
