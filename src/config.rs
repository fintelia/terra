use std::collections::BTreeMap;
use serde::{Serialize, Deserialize};

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
pub enum Registration {
    Vertex,
    Center,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
pub enum TextureFormat {
    R32F,
    Rgba8,
}


#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputKind {
    HeightMap,
    NormalMap,
    AlbedoMap,
    F32,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
pub enum Projection {
    #[serde(alias = "NAD83")]
    Nad83,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
pub enum DatasetFormat {
    #[serde(alias = "GridFloat+zip")]
    ZippedGridFloat,
}

#[derive(Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
pub struct NodeOutput {
    name: String,
    format: TextureFormat,
}

#[derive(Clone, Eq, PartialEq, Debug, Serialize, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
pub enum Node {
    Generated {
        shader: String,
        resolution: u64,
        kind: OutputKind,
        inputs: BTreeMap<String, String>,

        #[serde(default)]
        tiles: Option<u64>,

        #[serde(default)]
        center_registration: bool,
    },
    Dataset {
        url: String,
        projection: Projection,
        resolution: u64,
        format: DatasetFormat,
        bib: Option<String>,
        license: Option<String>,
    },
}
