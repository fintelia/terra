#![warn(non_snake_case)]

use std::str::FromStr;

use serde::de::Error;
use serde::{Deserialize, Deserializer};

#[derive(Deserialize, Debug)]
struct SpeedTreeRaw {
    VersionMajor: u8,
    VersionMinor: u8,
    UserData: String,
    Source: String,
    Wind: Wind,
    Objects: Objects,
}

#[derive(Deserialize, Debug)]
struct Wind {
    Bend: f32,
    BranchAmplitude: f32,
    LeafAmplitude: f32,
    DetailFrequency: f32,
    GlobalHeight: f32,
}

#[derive(Deserialize, Debug)]
struct Objects {
    Count: u32,
    LodNear: f32,
    LodFar: f32,
    BoundsMinX: f32,
    BoundsMinY: f32,
    BoundsMinZ: f32,
    BoundsMaxX: f32,
    BoundsMaxY: f32,
    BoundsMaxZ: f32,
    #[serde(rename = "Object", default)]
    objects: Vec<Object>,
}

#[derive(Deserialize, Debug)]
struct Object {
    ID: u32,
    Name: String,
    ParentId: Option<u32>,
    Points: Option<Points>,
    Vertices: Option<Vertices>,
}

#[derive(Deserialize, Debug)]
struct Points {
    Count: u32,
    X: List<f32>,
    Y: List<f32>,
    Z: List<f32>,
    LodX: List<f32>,
    LodY: List<f32>,
    LodZ: List<f32>,
}

#[derive(Deserialize, Debug)]
struct Vertices {
    Count: u32,
    NormalX: List<f32>,
    NormalY: List<f32>,
    NormalZ: List<f32>,
    BinormalX: List<f32>,
    BinormalY: List<f32>,
    BinormalZ: List<f32>,
    TangentX: List<f32>,
    TangentY: List<f32>,
    TangentZ: List<f32>,
    TexcoordU: List<f32>,
    TexcoordV: List<f32>,
    AO: List<f32>,
    VertexColorR: List<f32>,
    VertexColorG: List<f32>,
    VertexColorB: List<f32>,
}

#[derive(Deserialize, Debug)]
struct Triangles {
    Count: u32,
    Material: u32,
    PointIndices: List<u32>,
    VertexIndices: List<u32>,
}

#[derive(Debug)]
struct List<T>(Vec<T>);
impl<'de, T: Deserialize<'de> + FromStr> serde::Deserialize<'de> for List<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;

        match s.split(" ").map(T::from_str).collect() {
            Err(e) => Err(D::Error::custom("Failed to parse space seperated list")),
            Ok(values) => Ok(Self(values)),
        }
    }
}

#[repr(C)]
pub(crate) struct Vertex {
    position: [f32; 3],
    lod_position: [f32; 3],
    normal: [f32; 3],
    binormal: [f32; 3],
    texcoord: [f32; 2],
    ao: f32,
    color: u32,
}
pub(crate) struct SpeedTreeModel {
    lods: Vec<(Vec<Vertex>, Vec<u32>)>,
}

fn parse_xml(contents: &str) -> Result<SpeedTreeModel, anyhow::Error> {
    let t: SpeedTreeRaw = quick_xml::de::from_str(include_str!(
        "/home/jonathan/tmp/speedtree3/Oak_English_Sapling.xml"
    ))?;

    let root_objects = t;

    Ok(todo!())
}

#[cfg(test)]
mod tests {
    #[test]
    fn vertex_size() {
        assert_eq!(std::mem::size_of::<super::Vertex>(), 64);
    }
    #[test]
    fn parse_xml() {
        let t: super::SpeedTreeRaw = quick_xml::de::from_str(include_str!(
            "/home/jonathan/tmp/speedtree3/Oak_English_Sapling.xml"
        ))
        .unwrap();
    }
}
