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
    vertex: Option<Vec<u32>>,
    fragment: Option<Vec<u32>>,
    compute: Option<Vec<u32>>,
}
impl ShaderSet {
    pub fn simple(
        _: &mut ShaderDirectoryWatcher,
        vertex_source: ShaderSource,
        fragment_source: ShaderSource,
    ) -> Result<Self, failure::Error> {
        let vertex = Some(create_vertex_shader(&vertex_source.source.unwrap()).unwrap());
        let fragment = Some(create_fragment_shader(&fragment_source.source.unwrap()).unwrap());
        Ok(Self { vertex, fragment, compute: None })
    }
    pub fn compute_only(
        _: &mut ShaderDirectoryWatcher,
        compute_source: ShaderSource,
    ) -> Result<Self, failure::Error> {
        let compute = Some(create_compute_shader(&compute_source.source.unwrap()).unwrap());
        Ok(Self { vertex: None, fragment: None, compute })
    }
    pub fn refresh(&mut self, _: &mut ShaderDirectoryWatcher) -> bool {
        false
    }
    pub fn vertex(&self) -> &[u32] {
        self.vertex.as_ref().unwrap()
    }
    pub fn fragment(&self) -> &[u32] {
        self.fragment.as_ref().unwrap()
    }
    pub fn compute(&self) -> &[u32] {
        self.compute.as_ref().unwrap()
    }
}
