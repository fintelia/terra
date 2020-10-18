use notify;
use std::path::PathBuf;

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
    inner: ShaderSetInner,
}
impl ShaderSet {
    pub fn simple(
        _: &mut ShaderDirectoryWatcher,
        vertex_source: ShaderSource,
        fragment_source: ShaderSource,
    ) -> Result<Self, anyhow::Error> {
        Ok(Self {
            inner: ShaderSetInner::simple(
                vertex_source.source.unwrap(),
                fragment_source.source.unwrap(),
            )?,
        })
    }
    pub fn compute_only(
        _: &mut ShaderDirectoryWatcher,
        compute_source: ShaderSource,
    ) -> Result<Self, anyhow::Error> {
        Ok(Self { inner: ShaderSetInner::compute_only(compute_source.source.unwrap())? })
    }
    pub fn refresh(&mut self, _: &mut ShaderDirectoryWatcher) -> bool {
        false
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
