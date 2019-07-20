use std::collections::BTreeMap;
use serde::{Serialize, Deserialize};

#[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub enum Registration {
    Vertex,
    Center,
}

#[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub enum TextureFormat {
    R32F,
    Rgba8,
}
impl TextureFormat {
    pub fn bytes_per_pixel(&self) -> u32 {
        match self {
            TextureFormat::R32F => 4,
            TextureFormat::Rgba8 => 4,
        }
    }
}


#[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputKind {
    HeightMap,
    NormalMap,
    AlbedoMap,
    F32,
}

#[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub enum Projection {
    #[serde(alias = "NAD83")]
    Nad83,
}

#[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub enum DatasetFormat {
    #[serde(alias = "GridFloat+zip")]
    ZippedGridFloat,
}

// #[derive(Clone, Eq, PartialEq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
// pub struct NodeOutput {
//     name: String,
//     format: TextureFormat,
// }

#[derive(Clone, Eq, PartialEq, PartialOrd, Ord, Debug, Serialize, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
pub enum Node {
    Generated {
        shader: String,
        resolution: u32,
        kind: OutputKind,
        format: TextureFormat,
        inputs: BTreeMap<String, String>,
        cache_size: u16,

        #[serde(default)]
        tiles: Option<u64>,

        #[serde(default)]
        corner_registration: bool,
    },
    Dataset {
        url: String,
        projection: Projection,
        resolution: u32,
        format: DatasetFormat,
        bib: Option<String>,
        license: Option<String>,
        cache_size: u16,
    },
}

#[derive(Clone, Eq, PartialEq, Debug, Serialize, Deserialize)]
pub struct GraphFile {
    pub center: String,
    /// Map will be a square with this many sectors on each side.
    pub side_length_sectors: u16,
    pub nodes: BTreeMap<String, Node>,
    pub shaders: BTreeMap<String, String>,
}
