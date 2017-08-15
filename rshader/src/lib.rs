extern crate gfx;
extern crate notify;

pub mod static_shaders;
pub mod dynamic_shaders;

#[cfg(not(feature = "dynamic_shaders"))]
pub use static_shaders::*;
#[cfg(feature = "dynamic_shaders")]
pub use dynamic_shaders::*;

pub struct ShaderSource {
    pub source: Option<Vec<u8>>,
    pub filename: Option<String>,
}

#[macro_export]
#[cfg(not(feature = "dynamic_shaders"))]
macro_rules! load_shader_source {
    ($compile_time:expr, $runtime:expr) => {
        $crate::ShaderSource{
            source: Some(include_bytes!($compile_time).to_vec()),
            filename: None,
        }
    };
}

#[macro_export]
#[cfg(feature = "dynamic_shaders")]
macro_rules! load_shader_source {
    ($compile_time:expr, $runtime:expr) => {
        $crate::ShaderSource {
            source: None,
            filename: Some($runtime.to_string()),
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
