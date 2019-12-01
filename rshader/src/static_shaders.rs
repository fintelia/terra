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
    vertex: Option<SpirvShader>,
    fragment: Option<SpirvShader>,
    compute: Option<(SpirvShader, Vec<u8>)>,
}
impl ShaderSet {
    pub fn simple(
        _: &mut ShaderDirectoryWatcher,
        vertex_source: ShaderSource,
        fragment_source: ShaderSource,
    ) -> Result<Self, failure::Error> {
        let vertex = Some(create_vertex_shader(&vertex_source.source.unwrap()).unwrap());
        let fragment = Some(create_fragment_shader(&fragment_source.source.unwrap()).unwrap());
        Ok(Self {
            vertex,
            fragment,
            compute: None,
        })
    }
    pub fn compute_only(
        _: &mut ShaderDirectoryWatcher,
        compute_source: ShaderSource,
    ) -> Result<Self, failure::Error> {
        Ok(Self {
            vertex: None,
            fragment: None,
            compute: Some(create_compute_shader(&compute_source.source.unwrap()).unwrap()),
        })
    }
    pub fn refresh(&mut self, _: &mut ShaderDirectoryWatcher) -> bool {
        false
    }
    pub fn vertex(&self) -> &SpirvShader {
        self.vertex.as_ref().unwrap()
    }
    pub fn fragment(&self) -> &SpirvShader {
        self.fragment.as_ref().unwrap()
    }
    pub fn compute(&self) -> &[u8] {
        self.compute.as_ref().map(|v| &v.1).unwrap()
    }
}
