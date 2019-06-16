use failure::Error;
use memmap::MmapMut;
use petgraph::{graph::{NodeIndex, DiGraph}, visit::Topo};
use petgraph::visit::Walker;
use serde::{Serialize, Deserialize};
use std::collections::{BTreeMap, HashMap};
use std::fs;

mod description;

use description::{Node, OutputKind};

#[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub struct Sector(i32, i32);

#[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub struct LayerId(u32);

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
pub struct LayerHeader {
    resolution: u64,
    center_registration: bool,
    format: description::TextureFormat,

    sector_bytes: u64,
    sector_offsets: BTreeMap<Sector, u64>,
}

pub struct Layer {
    filename: String,

    header: LayerHeader,
    data: MmapMut,
}

pub struct Graph {
    config: BTreeMap<String, description::Node>,

    layer_ids: HashMap<String, LayerId>,
    layer_priorities: HashMap<LayerType, Vec<LayerId>>,
    layers: HashMap<LayerId, Layer>,

    order: Vec<LayerId>,

    sectors: u16,
}

impl Graph {
    #[allow(unused)]
    pub fn from_file(filename: &str, sectors: u16, latitude: i16, longitude: i16) -> Result<Graph, Error> {
        let file = fs::read_to_string(filename)?;
        let config: BTreeMap<String, description::Node> = toml::from_str(&file)?;

        let mut layer_ids = HashMap::new();
        let mut g = DiGraph::new();
        for name in config.keys() {
            let id = g.add_node(name);
            layer_ids.insert(name.to_owned(), LayerId(id.index() as u32));
        }
        for (name, node) in config.iter() {
            if let description::Node::Generated { ref inputs, .. } = node {
                let child = NodeIndex::new(layer_ids[name].0 as usize);
                for parent in inputs.values() {
                    g.add_edge(NodeIndex::new(layer_ids[parent].0 as usize), child, ());
                }
            }
        }

        let order: Vec<LayerId> = Topo::new(&g)
            .iter(&g)
            .map(|id| LayerId(id.index() as u32))
            .collect();

        let mut layer_priorities = HashMap::new();
        for id in order.iter().rev() {
            let ty = match config[g[NodeIndex::new(id.0 as usize)]] {
                Node::Generated { kind: OutputKind::HeightMap , .. } => LayerType::Heightmap,
                Node::Generated { kind: OutputKind::NormalMap , .. } => LayerType::Heightmap,
                Node::Generated { kind: OutputKind::AlbedoMap , .. } => LayerType::Albedo,
                _ => continue,
            };

            layer_priorities.entry(ty).or_insert(vec![]).push(*id);
        }

        println!("{:#?}", config);

        Ok(Graph {
            config,
            layer_ids,
            layer_priorities,
            layers: HashMap::new(),
            order,
            sectors,
        })
    }
}
