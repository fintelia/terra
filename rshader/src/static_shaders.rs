use std::path::PathBuf;

use gfx;
use failure::Error;
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

pub struct Shader<R: gfx::Resources> {
    shader_set: gfx::ShaderSet<R>,
}
impl<R: gfx::Resources> Shader<R> {
    pub fn simple<F: gfx::Factory<R>>(
        factory: &mut F,
        _: &mut ShaderDirectoryWatcher,
        vertex_source: ShaderSource,
        pixel_source: ShaderSource,
    ) -> Result<Self, Error> {
        let v = create_vertex_shader(factory, &vertex_source.source.unwrap()).unwrap();
        let f = create_pixel_shader(factory, &pixel_source.source.unwrap()).unwrap();
        Ok(Self {
            shader_set: gfx::ShaderSet::Simple(v, f),
        })
    }

    pub fn as_shader_set(&self) -> &gfx::ShaderSet<R> {
        &self.shader_set
    }
    pub fn refresh<F: gfx::Factory<R>>(
        &mut self,
        _: &mut F,
        _: &mut ShaderDirectoryWatcher,
    ) -> bool {
        false
    }
}
