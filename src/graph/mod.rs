use failure::{bail, format_err, Error};
use generic_array::{ArrayLength, GenericArray};
use gfx_hal::{pso::ShaderStageFlags, Backend};
use memmap::MmapMut;
use petgraph::visit::Walker;
use petgraph::{
    graph::{DiGraph, NodeIndex},
    visit::Topo,
};
use rendy::factory::Factory;
use rendy::shader::{ShaderSet, ShaderSetBuilder, SpirvShader};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};
use sled::{Db, IVec};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};
use std::path::PathBuf;
use xdg::BaseDirectories;

mod dataset;
mod description;

use dataset::{Dataset, DatasetDesc};
use description::{GraphFile, Node, OutputKind, TextureFormat};

#[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub struct Sector(i32, i32);

/// A LayerId is the Sha256 hash of the layer's LayerHeader.
#[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash, Debug)]
pub struct LayerId(GenericArray<u8, <Sha256 as Digest>::OutputSize>);
impl Serialize for LayerId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        hex::encode(self.0.as_slice()).serialize(serializer)
    }
}
impl<'de> Deserialize<'de> for LayerId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Error;
        <&str>::deserialize(deserializer).and_then(|s| {
            Ok(LayerId(GenericArray::clone_from_slice(
                &hex::decode(s).map_err(|e| D::Error::custom(e))?,
            )))
        })
    }
}

#[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub struct Tile(u16, u16);

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
pub enum LayerType {
    Heightmap,
    Normalmap,
    Albedo,
    /// layer that isn't used for rendering directly
    Intermediate,
}

#[derive(Serialize, Deserialize)]
pub struct LayerDesc {
    parents: BTreeMap<String, LayerId>,
    resolution: u64,
    corner_registration: bool,
    format: TextureFormat,
    sector_bytes: u64,
    shader: String,
    shader_name: String,
    center: String,
}

pub struct Layer<B: Backend> {
    desc: LayerDesc,
    filename: PathBuf,
    shader: ShaderSet<B>,
    data: MmapMut,
}
impl<B: Backend> Layer<B> {
    fn compute_sector_index(sector: Sector) -> u64 {
        let ax = if sector.0 >= 0 {
            sector.0
        } else {
            -sector.0 - 1
        } as u64;
        let ay = if sector.1 >= 0 {
            sector.1
        } else {
            -sector.1 - 1
        } as u64;
        let q = match (sector.0 >= 0, sector.1 >= 0) {
            (true, true) => 0,
            (false, true) => 1,
            (true, false) => 2,
            (false, false) => 3,
        };
        if ax > ay {
            (ax * ax + 2 * ay + 1) * 4 + q
        } else {
            (ay * ay + 2 * ax) * 4 + q
        }
    }
    fn compute_sector_offset(&self, sector: Sector) -> u64 {
        self.desc.sector_bytes * Self::compute_sector_index(sector)
    }
}

pub struct Graph<B: Backend> {
    config: GraphFile,
    xdg_dirs: BaseDirectories,

    layer_ids: HashMap<String, LayerId>,
    generated_layers: HashMap<LayerId, Layer<B>>,
    dataset_layers: HashMap<LayerId, Dataset>,

    order: Vec<LayerId>,
    priorities: HashMap<LayerType, Vec<LayerId>>,
}

impl<B: Backend> Graph<B> {
    #[allow(unused)]
    pub fn from_file(
        config_string: &str,
        xdg_dirs: BaseDirectories,
        factory: &mut Factory<B>,
    ) -> Result<Graph<B>, Error> {
        let config: GraphFile = toml::from_str(&config_string)?;
        let center = open_location_code::decode(&config.center)
            .map_err(|e| format_err!("{}", e))?
            .center;
        let center = (center.x(), center.y());

        let order: Vec<String> = {
            let mut ids = HashMap::new();
            let mut g = DiGraph::new();
            for name in config.nodes.keys() {
                ids.insert(name.to_owned(), g.add_node(name));
            }
            for (name, node) in config.nodes.iter() {
                if let description::Node::Generated { ref inputs, .. } = node {
                    let child = ids[name];
                    for parent in inputs.values() {
                        match ids.get(parent) {
                            Some(p) => g.add_edge(p.clone(), child, ()),
                            None => bail!("node.{} not found", parent),
                        };
                    }
                }
            }

            Topo::new(&g)
                .iter(&g)
                .map(|id| g.node_weight(id).unwrap().to_string())
                .collect()
        };

        let mut layer_ids = HashMap::new();
        let mut layer_descriptors = BTreeMap::new();
        let mut dataset_layers = HashMap::new();
        for name in &order {
            match &config.nodes[name] {
                Node::Dataset {
                    url,
                    resolution,
                    format,
                    bib,
                    license,
                    projection,
                    ..
                } => {
                    let desc = DatasetDesc {
                        url: url.to_owned(),
                        credentials: None,
                        projection: *projection,
                        resolution: *resolution,
                        file_format: *format,
                        texture_format: TextureFormat::R32F,
                    };
                    let desc_bytes = bincode::serialize(&desc)?;
                    let id = LayerId(Sha256::digest(&desc_bytes));
                    layer_ids.insert(name.to_owned(), id.clone());
                    let directory = xdg_dirs.create_cache_directory(format!(
                        "datasets/{}",
                        hex::encode(id.0.as_slice())
                    ))?;
                    fs::write(
                        directory.join("header.json"),
                        serde_json::to_string_pretty(&desc)?,
                    )?;
                    dataset_layers.insert(
                        id,
                        Dataset {
                            desc,
                            bib: bib.to_owned(),
                            license: license.to_owned(),
                            directory,
                        },
                    );
                }
                Node::Generated {
                    ref inputs,
                    resolution,
                    corner_registration,
                    format,
                    ref shader,
                    ..
                } => {
                    let desc = LayerDesc {
                        parents: {
                            let mut parents = BTreeMap::new();
                            for input in inputs.values() {
                                let id = layer_ids[input];
                                parents.insert(input.to_owned(), id);
                            }
                            parents
                        },
                        resolution: *resolution,
                        corner_registration: *corner_registration,
                        format: *format,
                        sector_bytes: resolution * resolution * format.bytes_per_pixel(),
                        shader: config
                            .shaders
                            .get(shader)
                            .ok_or(format_err!("Missing shader '{}'", shader))?
                            .to_owned(),
                        shader_name: shader.clone(),
                        center: config.center.clone(),
                    };
                    let desc_bytes = bincode::serialize(&desc)?;
                    let id = LayerId(Sha256::digest(&desc_bytes));
                    layer_ids.insert(name.to_owned(), id);
                    layer_descriptors.insert(name.to_owned(), desc);
                }
            };
        }

        let mut priorities = HashMap::new();
        for name in order.iter().rev() {
            let ty = match config.nodes[name] {
                Node::Generated {
                    kind: OutputKind::HeightMap,
                    ..
                } => LayerType::Heightmap,
                Node::Generated {
                    kind: OutputKind::NormalMap,
                    ..
                } => LayerType::Heightmap,
                Node::Generated {
                    kind: OutputKind::AlbedoMap,
                    ..
                } => LayerType::Albedo,
                _ => continue,
            };

            priorities.entry(ty).or_insert(vec![]).push(layer_ids[name]);
        }

        let mut glsl_compiler =
            shaderc::Compiler::new().ok_or(format_err!("Shader compiler init failed"))?;

        let mut generated_layers = HashMap::new();
        for (name, desc) in layer_descriptors {
            let id = layer_ids[&name].to_owned();
            let hash = hex::encode(id.0.as_slice());
            let header_filename =
                xdg_dirs.place_cache_file(format!("generated/{}.header", &hash))?;
            let data_filename = xdg_dirs.place_cache_file(format!("generated/{}.data", &hash))?;

            fs::write(header_filename, serde_json::to_string_pretty(&desc)?);
            let mut file = OpenOptions::new()
                .create(true)
                .read(true)
                .write(true)
                .open(&data_filename)?;

            // TODO: Seeking past end is implementation defined. Use a cross platform option instead.
            let target_size = desc.sector_bytes
                * config.side_length_sectors as u64
                * config.side_length_sectors as u64;
            file.seek(SeekFrom::Start(target_size))?;
            file.write_all(&[0u8])?;
            file.seek(SeekFrom::Start(0))?;

            let data = unsafe { MmapMut::map_mut(&file)? };

            let spirv = glsl_compiler
                .compile_into_spirv(
                    &desc.shader,
                    shaderc::ShaderKind::Compute,
                    &desc.shader_name,
                    "main",
                    None,
                )?
                .as_binary_u8()
                .to_vec();
            let shader = SpirvShader::new(spirv, ShaderStageFlags::COMPUTE, "main");
            let shader = ShaderSetBuilder::default()
                .with_compute(&shader)?
                .build(&factory, Default::default())?;

            generated_layers.insert(
                id,
                Layer {
                    desc,
                    filename: data_filename,
                    shader,
                    data,
                },
            );
        }

        Ok(Graph {
            config,
            xdg_dirs,
            order: order.iter().map(|name| layer_ids[name]).collect(),
            layer_ids,
            priorities,
            generated_layers,
            dataset_layers,
        })
    }

    fn generate(&mut self, sector: Sector, layer: LayerId) {}
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn compute_sector_index() {
        assert_eq!(Layer::compute_sector_index(Sector(0, 0)), 0 * 4);
        assert_eq!(Layer::compute_sector_index(Sector(1, 1)), 3 * 4);
        assert_eq!(Layer::compute_sector_index(Sector(1, 2)), 6 * 4);
        assert_eq!(Layer::compute_sector_index(Sector(2, 3)), 13 * 4);
        assert_eq!(Layer::compute_sector_index(Sector(4, 1)), 19 * 4);

        assert_eq!(Layer::compute_sector_index(Sector(-3, -1)), 5 * 4 + 3);
        assert_eq!(Layer::compute_sector_index(Sector(-3, -4)), 13 * 4 + 3);
    }
}
