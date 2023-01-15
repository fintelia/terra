use std::{borrow::Cow, collections::HashMap, num::NonZeroU8};

use crate::{
    billboards::Models,
    cache::{Levels, MeshType, TileCache, LAYERS_BY_NAME},
    mapfile::MapFile,
};
use types::MAX_QUADTREE_LEVEL;
use vec_map::VecMap;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct DrawIndexedIndirect {
    pub(crate) vertex_count: u32,   // The number of vertices to draw.
    pub(crate) instance_count: u32, // The number of instances to draw.
    pub(crate) base_index: u32,     // The base index within the index buffer.
    pub(crate) vertex_offset: i32, // The value added to the vertex index before indexing into the vertex buffer.
    pub(crate) base_instance: u32, // The instance ID of the first instance to draw.
}
unsafe impl bytemuck::Pod for DrawIndexedIndirect {}
unsafe impl bytemuck::Zeroable for DrawIndexedIndirect {}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct GlobalUniformBlock {
    pub view_proj: mint::ColumnMatrix4<f32>,
    pub view_proj_inverse: mint::ColumnMatrix4<f32>,
    pub shadow_view_proj: mint::ColumnMatrix4<f32>,
    pub frustum_planes: [[f32; 4]; 5],
    pub camera: [f32; 3],
    pub screen_width: f32,
    pub sun_direction: [f32; 3],
    pub screen_height: f32,
    pub sidereal_time: f32,
    pub exposure: f32,
    pub _padding: [f32; 2],
}
unsafe impl bytemuck::Pod for GlobalUniformBlock {}
unsafe impl bytemuck::Zeroable for GlobalUniformBlock {}

pub(crate) fn texture_from_ktx2_bytes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bytes: &[u8],
    label: &str,
) -> Result<wgpu::Texture, anyhow::Error> {
    let reader = ktx2::Reader::new(bytes)?;

    let header = reader.header();
    assert_eq!(header.supercompression_scheme, None);

    let format = match header.format {
        Some(ktx2::Format::R8_UNORM) => wgpu::TextureFormat::R8Unorm,
        Some(ktx2::Format::R8G8_UNORM) => wgpu::TextureFormat::Rg8Unorm,
        Some(ktx2::Format::R8G8B8A8_UNORM) => wgpu::TextureFormat::Rgba8Unorm,
        Some(ktx2::Format::R32G32B32A32_SFLOAT) => wgpu::TextureFormat::Rgba32Float,
        _ => unimplemented!("Unsupported format: {:?}", header.format),
    };
    let format_info = format.describe();
    assert_eq!(format_info.block_dimensions.0, format_info.block_dimensions.1);

    let data = reader.levels().flatten().copied().collect::<Vec<_>>();
    Ok(device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some(&format!("texture.{}", label)),
            size: wgpu::Extent3d {
                width: header.pixel_width,
                height: header.pixel_height,
                depth_or_array_layers: if header.pixel_depth > 1 {
                    header.pixel_depth
                } else {
                    header.layer_count.max(1) * header.face_count
                },
            },
            mip_level_count: header.level_count.max(1),
            sample_count: 1,
            dimension: if header.pixel_depth > 1 {
                wgpu::TextureDimension::D3
            } else if header.pixel_height > 1 {
                wgpu::TextureDimension::D2
            } else {
                wgpu::TextureDimension::D1
            },
            format,
            usage: if format_info.is_compressed() {
                wgpu::TextureUsages::TEXTURE_BINDING
            } else {
                wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING
            },
        },
        &data,
    ))
}

pub(crate) struct GpuState {
    pub tile_cache: VecMap<Vec<(wgpu::Texture, wgpu::TextureView)>>,

    pub mesh_index: wgpu::Buffer,
    pub mesh_storage: VecMap<wgpu::Buffer>,
    pub mesh_indirect: wgpu::Buffer,
    pub mesh_bounding: wgpu::Buffer,

    pub model_storage: wgpu::Buffer,
    pub model_indices: wgpu::Buffer,

    pub globals: wgpu::Buffer,
    pub generate_uniforms: wgpu::Buffer,
    pub starfield: wgpu::Buffer,

    pub nodes: wgpu::Buffer,
    pub frame_nodes: wgpu::Buffer,

    noise: (wgpu::Texture, wgpu::TextureView),
    sky: (wgpu::Texture, wgpu::TextureView),
    cloudcover: (wgpu::Texture, wgpu::TextureView),
    transmittance: (wgpu::Texture, wgpu::TextureView),
    inscattering: (wgpu::Texture, wgpu::TextureView),
    skyview: (wgpu::Texture, wgpu::TextureView),

    pub models_albedo: (wgpu::Texture, wgpu::TextureView),

    pub billboards_albedo: (wgpu::Texture, wgpu::TextureView),
    pub billboards_normals: (wgpu::Texture, wgpu::TextureView),
    pub billboards_depth: (wgpu::Texture, wgpu::TextureView),
    pub billboards_ao: (wgpu::Texture, wgpu::TextureView),
    pub topdown_albedo: (wgpu::Texture, wgpu::TextureView),
    pub topdown_normals: (wgpu::Texture, wgpu::TextureView),
    pub topdown_depth: (wgpu::Texture, wgpu::TextureView),
    pub topdown_ao: (wgpu::Texture, wgpu::TextureView),

    pub shadowmap: (wgpu::Texture, wgpu::TextureView),

    ground_albedo: (wgpu::Texture, wgpu::TextureView),
    nearest: wgpu::Sampler,
    linear: wgpu::Sampler,
    linear_wrap: wgpu::Sampler,
    shadow_sampler: wgpu::Sampler,
}
impl GpuState {
    pub(crate) async fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mapfile: &MapFile,
        cache: &TileCache,
        models: &Models,
    ) -> Result<Self, anyhow::Error> {
        let with_view = |name: &'static str, t: wgpu::Texture| {
            let view = t.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("texture.{}.view", name)),
                ..Default::default()
            });
            (t, view)
        };

        let from_ktx2 = |(filename, bytes):  (&'static str,Vec<u8>)|
            -> Result<(wgpu::Texture, wgpu::TextureView), anyhow::Error>
        {
            Ok(with_view(filename, texture_from_ktx2_bytes(device, queue, &bytes, filename)?))
        };

        let (model_storage, model_indices) = models.make_buffers(device);

        async fn download(mapfile: &MapFile, name: &'static str) -> (&'static str, Vec<u8>) {
            (name, mapfile.read_asset(name).await.expect(&format!("failed to download {}", name)))
        }
        let (noise, sky, cloudcover, transmittance, inscattering, ground_albedo) = tokio::join!(
            download(mapfile, "noise.ktx2"),
            download(mapfile, "sky.ktx2"),
            download(mapfile, "cloudcover.ktx2"),
            download(mapfile, "transmittance.ktx2"),
            download(mapfile, "inscattering.ktx2"),
            download(mapfile, "ground_albedo.ktx2"),
        );

        Ok(GpuState {
            noise: from_ktx2(noise)?,
            sky: from_ktx2(sky)?,
            cloudcover: from_ktx2(cloudcover)?,
            transmittance: from_ktx2(transmittance)?,
            inscattering: from_ktx2(inscattering)?,
            ground_albedo: from_ktx2(ground_albedo)?,

            skyview: with_view(
                "skyview",
                device.create_texture(&wgpu::TextureDescriptor {
                    size: wgpu::Extent3d { width: 128, height: 128, depth_or_array_layers: 1 },
                    format: wgpu::TextureFormat::Rgba16Float,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    usage: wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    label: Some("texture.skyview"),
                }),
            ),
            models_albedo: with_view("models.albedo", models.make_models_albedo(device, queue)?),
            billboards_albedo: with_view(
                "billboards.albedo",
                models.make_billboards_albedo(device),
            ),
            billboards_normals: with_view(
                "billboards.normals",
                models.make_billboards_normals(device),
            ),
            billboards_depth: with_view("billboards.depth", models.make_billboards_depth(device)),
            billboards_ao: with_view("billboards.ao", models.make_billboards_ao(device)),
            topdown_albedo: with_view("topdown.albedo", models.make_topdown_albedo(device)),
            topdown_normals: with_view("topdown.normals", models.make_topdown_normals(device)),
            topdown_depth: with_view("topdown.depth", models.make_topdown_depth(device)),
            topdown_ao: with_view("topdown.ao", models.make_topdown_ao(device)),

            shadowmap: with_view(
                "shadowmap",
                device.create_texture(&wgpu::TextureDescriptor {
                    size: wgpu::Extent3d { width: 8192, height: 8192, depth_or_array_layers: 1 },
                    format: wgpu::TextureFormat::Depth24Plus,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    label: Some("texture.shadowmap"),
                }),
            ),

            tile_cache: cache.make_gpu_tile_cache(device),
            mesh_index: cache.make_gpu_mesh_index(device),
            mesh_storage: cache.make_gpu_mesh_storage(device),
            mesh_indirect: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                contents: &vec![
                    0;
                    std::mem::size_of::<DrawIndexedIndirect>()
                        * cache.total_mesh_entries()
                ],
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_DST,
                label: Some("buffer.mesh_indirect"),
            }),
            mesh_bounding: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                contents: &vec![0; 16 * cache.total_mesh_entries()],
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_DST,
                label: Some("buffer.mesh_bounding"),
            }),
            model_storage,
            model_indices,
            starfield: {
                let mut stars = vec![0.0f32; 4 * 9096];
                bytemuck::cast_slice_mut(&mut stars)
                    .copy_from_slice(include_bytes!("../assets/stars.bin"));
                for star in stars.chunks_mut(4) {
                    let (gal_lat, gal_long) = (star[0] as f64, star[1] as f64);
                    star[0] = astro::coords::dec_frm_gal(gal_long, gal_lat) as f32;
                    star[1] = astro::coords::asc_frm_gal(gal_long, gal_lat) as f32;
                }
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("buffer.starfield"),
                    contents: bytemuck::cast_slice(&stars),
                    usage: wgpu::BufferUsages::STORAGE,
                })
            },
            globals: device.create_buffer(&wgpu::BufferDescriptor {
                size: std::mem::size_of::<GlobalUniformBlock>() as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
                label: Some("buffer.globals"),
                mapped_at_creation: false,
            }),
            generate_uniforms: device.create_buffer(&wgpu::BufferDescriptor {
                size: 256 * 1024,
                usage: wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::STORAGE,
                label: Some("buffer.generate.tiles"),
                mapped_at_creation: false,
            }),
            frame_nodes: device.create_buffer(&wgpu::BufferDescriptor {
                size: 4 * Levels::base_slot(MAX_QUADTREE_LEVEL + 1) as u64,
                usage: wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::STORAGE,
                label: Some("buffer.frame_nodes"),
                mapped_at_creation: false,
            }),
            nodes: device.create_buffer(&wgpu::BufferDescriptor {
                size: 1024 * Levels::base_slot(MAX_QUADTREE_LEVEL + 1) as u64,
                usage: wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::STORAGE,
                label: Some("buffer.nodes"),
                mapped_at_creation: false,
            }),
            nearest: device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                mipmap_filter: wgpu::FilterMode::Nearest,
                label: Some("sampler.nearest"),
                ..Default::default()
            }),
            linear: device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Linear,
                anisotropy_clamp: Some(NonZeroU8::new(4).unwrap()),
                label: Some("sampler.linear"),
                ..Default::default()
            }),
            linear_wrap: device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                address_mode_w: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Nearest,
                label: Some("sampler.linear_wrap"),
                ..Default::default()
            }),
            shadow_sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Nearest,
                label: Some("sampler.shadow"),
                compare: Some(wgpu::CompareFunction::GreaterEqual),
                ..Default::default()
            }),
        })
    }

    pub(crate) fn bind_group_for_shader(
        &self,
        device: &wgpu::Device,
        shader: &rshader::ShaderSet,
        buffers: HashMap<Cow<str>, (bool, wgpu::BindingResource)>,
        image_views: HashMap<Cow<str>, &wgpu::TextureView>,
        group_name: &str,
    ) -> (wgpu::BindGroup, wgpu::BindGroupLayout) {
        let mut layout_descriptor_entries = shader.layout_descriptor().entries.to_vec();

        let mut buffers = buffers;
        let mut image_views = image_views;
        //let mut samplers = HashMap::new();
        for (name, layout) in shader.desc_names().iter().zip(layout_descriptor_entries.iter()) {
            let name = &**name.as_ref().unwrap();
            match layout.ty {
                wgpu::BindingType::StorageTexture { .. } | wgpu::BindingType::Texture { .. } => {
                    if !image_views.contains_key(name) {
                        image_views.insert(
                            name.into(),
                            match name {
                                "noise" => &self.noise.1,
                                "sky" => &self.sky.1,
                                "cloudcover" => &self.cloudcover.1,
                                "transmittance" => &self.transmittance.1,
                                "inscattering" => &self.inscattering.1,
                                "skyview" => &self.skyview.1,
                                "models_albedo" => &self.models_albedo.1,
                                "billboards_albedo" => &self.billboards_albedo.1,
                                "billboards_normals" => &self.billboards_normals.1,
                                "billboards_ao" => &self.billboards_ao.1,
                                "billboards_depth" => &self.billboards_depth.1,
                                "topdown_albedo" => &self.topdown_albedo.1,
                                "topdown_normals" => &self.topdown_normals.1,
                                "shadowmap" => &self.shadowmap.1,
                                "ground_albedo" => &self.ground_albedo.1,
                                _ => match name.rsplit_once(char::is_numeric) {
                                    Some((name, suffix)) => {
                                        &self.tile_cache[LAYERS_BY_NAME[name]]
                                            [suffix.parse::<usize>().unwrap()]
                                        .1
                                    }
                                    None => &self.tile_cache[LAYERS_BY_NAME[name]][0].1,
                                },
                            },
                        );
                    }
                }
                wgpu::BindingType::Buffer { .. } => {
                    if !buffers.contains_key(name) {
                        let buffer = match name {
                            "mesh_indirect" => &self.mesh_indirect,
                            "mesh_bounding" => &self.mesh_bounding,
                            "model_storage" => &self.model_storage,
                            "grass_storage" => &self.mesh_storage[MeshType::Grass],
                            "tree_billboards_storage" => {
                                &self.mesh_storage[MeshType::TreeBillboards]
                            }
                            "globals" => &self.globals,
                            "frame_nodes" => &self.frame_nodes,
                            "nodes" => &self.nodes,
                            "starfield" => &self.starfield,
                            _ => unreachable!("unrecognized storage buffer: {}", name),
                        };
                        let resource = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            size: None,
                            offset: 0,
                        });
                        buffers.insert(name.into(), (false, resource));
                    }
                }
                wgpu::BindingType::Sampler { .. } => {}
            }
        }

        let mut bindings = Vec::new();
        for (name, layout) in shader.desc_names().iter().zip(layout_descriptor_entries.iter_mut()) {
            let name = &**name.as_ref().unwrap();
            bindings.push(wgpu::BindGroupEntry {
                binding: layout.binding,
                resource: match layout.ty {
                    wgpu::BindingType::Sampler(ref mut binding_type) => {
                        wgpu::BindingResource::Sampler(match name {
                            "nearest" => {
                                *binding_type = wgpu::SamplerBindingType::NonFiltering;
                                &self.nearest
                            }
                            "linear" | "linearsamp" => &self.linear,
                            "linear_wrap" => &self.linear_wrap,
                            "shadow_sampler" => &self.shadow_sampler,
                            _ => unreachable!("unrecognized sampler: {}", name),
                        })
                    }
                    wgpu::BindingType::StorageTexture { .. } => {
                        wgpu::BindingResource::TextureView(&image_views[name])
                    }
                    wgpu::BindingType::Texture { ref mut sample_type, .. } => {
                        match name {
                            "transmittance" | "inscattering" | "displacements" => {
                                *sample_type = wgpu::TextureSampleType::Float { filterable: false }
                            }
                            "heightmaps" | "heightmaps_in" => {
                                *sample_type = wgpu::TextureSampleType::Uint;
                            }
                            "shadowmap" => {
                                *sample_type = wgpu::TextureSampleType::Depth;
                            }
                            _ => {}
                        }
                        wgpu::BindingResource::TextureView(&image_views[name])
                    }
                    wgpu::BindingType::Buffer { ref mut has_dynamic_offset, .. } => {
                        let (d, ref buf) = buffers[name];
                        *has_dynamic_offset = d;
                        buf.clone()
                    }
                },
            });
        }

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &layout_descriptor_entries,
            label: Some(&format!("layout.{}", group_name)),
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &*bindings,
            label: Some(&format!("bindgroup.{}", group_name)),
        });

        (bind_group, bind_group_layout)
    }
}
