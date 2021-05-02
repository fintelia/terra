use naga::{
    Binding, ImageClass, ImageDimension, ScalarKind, StorageAccess, StorageClass, StorageFormat,
    TypeInner, VectorSize,
};

fn reflect(
    module: &naga::Module,
) -> Result<
    (Vec<wgpu::VertexAttribute>, Vec<Option<String>>, Vec<wgpu::BindGroupLayoutEntry>),
    anyhow::Error,
> {
    let mut attribute_offset = 0;

    let mut names = Vec::new();
    let mut bindings = Vec::new();
    let mut attributes = Vec::new();

    let mut visibility = wgpu::ShaderStage::empty();

    for entry in module.entry_points.iter() {
        let stage = match entry.stage {
            naga::ShaderStage::Vertex => wgpu::ShaderStage::VERTEX,
            naga::ShaderStage::Fragment => wgpu::ShaderStage::FRAGMENT,
            naga::ShaderStage::Compute => wgpu::ShaderStage::COMPUTE,
        };

        visibility |= stage; // right now say all globals are visible in all stages.

        if stage == wgpu::ShaderStage::VERTEX {
            for input in entry.function.arguments.iter() {
                let shader_location = match *input.binding.as_ref().expect("struct inputs not supported") {
                    Binding::BuiltIn(_) => continue,
                    Binding::Location { location, .. } => location,
                };

                let (format, nbytes) = match module.types[input.ty].inner {
                    TypeInner::Scalar { kind: ScalarKind::Sint, width: 4, .. } => {
                        (wgpu::VertexFormat::Sint32, 4)
                    }
                    TypeInner::Scalar { kind: ScalarKind::Uint, width: 4, .. } => {
                        (wgpu::VertexFormat::Uint32, 4)
                    }
                    TypeInner::Scalar { kind: ScalarKind::Float, width: 4, .. } => {
                        (wgpu::VertexFormat::Float32, 4)
                    }
                    TypeInner::Vector {
                        size: VectorSize::Bi,
                        kind: ScalarKind::Sint,
                        width: 8,
                    } => (wgpu::VertexFormat::Sint32x2, 8),
                    TypeInner::Vector {
                        size: VectorSize::Bi,
                        kind: ScalarKind::Uint,
                        width: 8,
                    } => (wgpu::VertexFormat::Uint32x2, 8),
                    TypeInner::Vector {
                        size: VectorSize::Bi,
                        kind: ScalarKind::Float,
                        width: 8,
                    } => (wgpu::VertexFormat::Float32x2, 8),
                    TypeInner::Vector {
                        size: VectorSize::Tri,
                        kind: ScalarKind::Sint,
                        width: 12,
                    } => (wgpu::VertexFormat::Sint32x3, 12),
                    TypeInner::Vector {
                        size: VectorSize::Tri,
                        kind: ScalarKind::Uint,
                        width: 12,
                    } => (wgpu::VertexFormat::Uint32x3, 12),
                    TypeInner::Vector {
                        size: VectorSize::Tri,
                        kind: ScalarKind::Float,
                        width: 12,
                    } => (wgpu::VertexFormat::Float32x3, 12),
                    TypeInner::Vector {
                        size: VectorSize::Quad,
                        kind: ScalarKind::Sint,
                        width: 16,
                    } => (wgpu::VertexFormat::Sint32x4, 16),
                    TypeInner::Vector {
                        size: VectorSize::Quad,
                        kind: ScalarKind::Uint,
                        width: 16,
                    } => (wgpu::VertexFormat::Uint32x4, 16),
                    TypeInner::Vector {
                        size: VectorSize::Quad,
                        kind: ScalarKind::Float,
                        width: 16,
                    } => (wgpu::VertexFormat::Float32x4, 16),

                    _ => unimplemented!(),
                };

                attributes.push(wgpu::VertexAttribute {
                    offset: attribute_offset,
                    format,
                    shader_location,
                });
                attribute_offset += nbytes;
            }
        }
    }

    for (_, global) in module.global_variables.iter() {
        let (set, binding) = match global.binding {
            None => continue,
            Some(ref b) => (b.group, b.binding),
        };
        assert_eq!(set, 0);

        let ty = match global.class {
            StorageClass::Handle => match module.types[global.ty].inner {
                TypeInner::Sampler { comparison } => {
                    wgpu::BindingType::Sampler { filtering: true, comparison }
                }
                TypeInner::Image { dim, arrayed, class } => {
                    let view_dimension = match (dim, arrayed) {
                        (ImageDimension::D2, false) => wgpu::TextureViewDimension::D2,
                        (ImageDimension::D2, true) => wgpu::TextureViewDimension::D2Array,
                        (ImageDimension::D3, false) => wgpu::TextureViewDimension::D3,
                        _ => unimplemented!(),
                    };
                    let access = if !global.storage_access.contains(StorageAccess::STORE) {
                        wgpu::StorageTextureAccess::ReadOnly
                    } else if !global.storage_access.contains(StorageAccess::LOAD) {
                        wgpu::StorageTextureAccess::WriteOnly
                    } else {
                        wgpu::StorageTextureAccess::ReadWrite
                    };
                    match class {
                        ImageClass::Storage(f) => wgpu::BindingType::StorageTexture {
                            view_dimension,
                            access,
                            format: match f {
                                StorageFormat::R32Float => wgpu::TextureFormat::R32Float,
                                StorageFormat::Rg32Float => wgpu::TextureFormat::Rg32Float,
                                StorageFormat::Rgba32Float => wgpu::TextureFormat::Rgba32Float,
                                StorageFormat::R32Uint => wgpu::TextureFormat::R32Uint,
                                StorageFormat::Rg32Uint => wgpu::TextureFormat::Rg32Uint,
                                StorageFormat::Rgba32Uint => wgpu::TextureFormat::Rgba32Uint,
                                StorageFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
                                _ => unimplemented!("component type {:?}", f),
                            },
                        },
                        ImageClass::Sampled {kind , multi} => wgpu::BindingType::Texture {
                            multisampled: multi,
                            view_dimension,
                            sample_type: match kind {
                                ScalarKind::Float => wgpu::TextureSampleType::Float { filterable: true },
                                ScalarKind::Uint => wgpu::TextureSampleType::Uint,
                                ScalarKind::Sint => wgpu::TextureSampleType::Sint,
                                ScalarKind::Bool => unreachable!(""),
                            }
                        },
                        ImageClass::Depth => unimplemented!(),
                    }
                }
                _ => unimplemented!(),
            },
            StorageClass::Uniform => wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            StorageClass::Storage => wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            v => unimplemented!("{:?}", v),
        };

        names.push(global.name.clone());
        bindings.push(wgpu::BindGroupLayoutEntry { binding, visibility, ty, count: None });
    }

    Ok((attributes, names, bindings))
}
