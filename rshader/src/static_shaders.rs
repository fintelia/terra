use std::path::PathBuf;

use notify;

use super::*;

pub struct ShaderDirectoryWatcher {}
impl ShaderDirectoryWatcher {
    pub fn new<P>(_: P) -> Result<Self, notify::Error>
    where
        PathBuf: From<P>,
    {
        Ok(Self {})
    }
}

pub struct ShaderSet {
    vertex: SpirvShader,
    fragment: SpirvShader,
}
impl ShaderSet {
    pub fn simple(
        _: &mut ShaderDirectoryWatcher,
        vertex_source: ShaderSource,
        pixel_source: ShaderSource,
    ) -> Result<Self, failure::Error> {
        let vertex = create_vertex_shader(&vertex_source.source.unwrap()).unwrap();
        let fragment = create_pixel_shader(&pixel_source.source.unwrap()).unwrap();
        Ok(Self { vertex, fragment })
    }
    pub fn refresh(&mut self, _: &mut ShaderDirectoryWatcher) -> bool {
        false
    }
    pub fn vertex(&self) -> &SpirvShader {
        &self.vertex
    }
    pub fn fragment(&self) -> &SpirvShader {
        &self.fragment
    }
}
