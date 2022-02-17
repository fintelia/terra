use std::{borrow::Cow, collections::HashMap};

use crate::{
    billboards::Models,
    cache::{MeshType, TileCache, LAYERS_BY_NAME},
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
    pub frustum_planes: [[f32; 4]; 5],
    pub camera: [f32; 4],
    pub sun_direction: [f32; 4],
}
unsafe impl bytemuck::Pod for GlobalUniformBlock {}
unsafe impl bytemuck::Zeroable for GlobalUniformBlock {}

pub(crate) struct GpuState {
    pub tile_cache: VecMap<(wgpu::Texture, wgpu::TextureView)>,

    pub mesh_storage: VecMap<wgpu::Buffer>,
    pub mesh_indirect: wgpu::Buffer,
    pub mesh_bounding: wgpu::Buffer,

    pub model_storage: wgpu::Buffer,
    pub model_indices: wgpu::Buffer,

    pub globals: wgpu::Buffer,
    pub generate_uniforms: wgpu::Buffer,

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

    //ground_albedo: (wgpu::Texture, wgpu::TextureView),
    nearest: wgpu::Sampler,
    linear: wgpu::Sampler,
    linear_wrap: wgpu::Sampler,
}
impl GpuState {
    pub(crate) fn new(
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

        let (model_storage, model_indices) = models.make_buffers(device);

        Ok(GpuState {
            noise: with_view("noise", mapfile.read_texture(device, queue, "noise")?),
            sky: with_view("sky", mapfile.read_texture(device, queue, "sky")?),
            cloudcover: with_view("sky", mapfile.read_texture(device, queue, "cloudcover")?),
            transmittance: with_view(
                "transmittance",
                mapfile.read_texture(device, queue, "transmittance")?,
            ),
            inscattering: with_view(
                "inscattering",
                mapfile.read_texture(device, queue, "inscattering")?,
            ),
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
            models_albedo: with_view("models.albedo", models.make_models_albedo(device, queue)),
            billboards_albedo: with_view("billboards.albedo", models.make_billboards_albedo(device)),
            billboards_normals: with_view("billboards.normals", models.make_billboards_normals(device)),
            billboards_depth: with_view("billboards.depth", models.make_billboards_depth(device)),
            billboards_ao: with_view("billboards.ao", models.make_billboards_ao(device)),
            topdown_albedo: with_view("topdown.albedo", models.make_topdown_albedo(device)),
            topdown_normals: with_view("topdown.normals", models.make_topdown_normals(device)),
            topdown_depth: with_view("topdown.depth", models.make_topdown_depth(device)),
            topdown_ao: with_view("topdown.ao", models.make_topdown_ao(device)),
            // ground_albedo: with_view(
            //     "ground_albedo",
            //     mapfile.read_texture(device, queue, "ground_albedo")?,
            // ),
            tile_cache: cache.make_gpu_tile_cache(device),
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
                size: 4 * TileCache::base_slot(MAX_QUADTREE_LEVEL + 1) as u64,
                usage: wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::STORAGE,
                label: Some("buffer.frame_nodes"),
                mapped_at_creation: false,
            }),
            nodes: device.create_buffer(&wgpu::BufferDescriptor {
                size: 1024 * TileCache::base_slot(MAX_QUADTREE_LEVEL + 1) as u64,
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
                mipmap_filter: wgpu::FilterMode::Nearest,
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
                                // "ground_albedo" => &self.ground_albedo.1,
                                _ => &self.tile_cache[LAYERS_BY_NAME[name]].1,
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
                            "linear" => &self.linear,
                            "linear_wrap" => &self.linear_wrap,
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
