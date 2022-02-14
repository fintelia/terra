#![warn(non_snake_case)]

use std::collections::HashMap;
use std::str::FromStr;

use serde::de::Error;
use serde::{Deserialize, Deserializer};
use std::ops::Deref;
use std::ops::Range;

#[derive(Deserialize, Debug)]
#[serde(rename_all = "PascalCase")]
#[allow(unused)]
struct SpeedTreeRaw {
    version_major: u8,
    version_minor: u8,
    user_data: String,
    source: String,
    wind: Wind,
    objects: Objects,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "PascalCase")]
#[allow(unused)]
struct Wind {
    bend: f32,
    branch_amplitude: f32,
    leaf_amplitude: f32,
    detail_frequency: f32,
    global_height: f32,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "PascalCase")]
#[allow(unused)]
struct Objects {
    count: u32,
    lod_near: f32,
    lod_far: f32,
    bounds_min_x: f32,
    bounds_min_y: f32,
    bounds_min_z: f32,
    bounds_max_x: f32,
    bounds_max_y: f32,
    bounds_max_z: f32,
    #[serde(rename = "Object", default)]
    objects: Vec<Object>,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "PascalCase")]
struct Object {
    #[serde(rename = "ID")]
    id: u32,
    name: String,
    #[serde(rename = "ParentID")]
    parent_id: Option<u32>,
    points: Option<Points>,
    vertices: Option<Vertices>,
    triangles: Option<Triangles>,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "PascalCase")]
struct Points {
    x: List<f32>,
    y: List<f32>,
    z: List<f32>,
    lod_x: List<f32>,
    lod_y: List<f32>,
    lod_z: List<f32>,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "PascalCase")]
#[allow(unused)]
struct Vertices {
    count: u32,
    normal_x: List<f32>,
    normal_y: List<f32>,
    normal_z: List<f32>,
    binormal_x: List<f32>,
    binormal_y: List<f32>,
    binormal_z: List<f32>,
    tangent_x: List<f32>,
    tangent_y: List<f32>,
    tangent_z: List<f32>,
    texcoord_u: List<f32>,
    texcoord_v: List<f32>,
    #[serde(rename = "AO")]
    ao: List<f32>,
    vertex_color_r: List<f32>,
    vertex_color_g: List<f32>,
    vertex_color_b: List<f32>,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "PascalCase")]
struct Triangles {
    count: u32,
    material: u32,
    point_indices: List<u32>,
    vertex_indices: List<u32>,
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
            Err(_) => Err(D::Error::custom("Failed to parse space seperated list")),
            Ok(values) => Ok(Self(values)),
        }
    }
}
impl<T> Deref for List<T> {
    type Target = Vec<T>;
    fn deref(&self) -> &Vec<T> {
        &self.0
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub(crate) struct Vertex {
    position: [f32; 3],
    ao: f32,

    lod_position: [f32; 3],
    color: u32,
    normal: [f32; 3],
    texcoord_u: f32,
    binormal: [f32; 3],
    texcoord_v: f32,
}
unsafe impl bytemuck::Pod for Vertex {}
unsafe impl bytemuck::Zeroable for Vertex {}

pub(crate) struct SpeedTreeModel {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub lods: Vec<Range<u32>>,
}

pub(crate) fn parse_xml(contents: &str) -> Result<SpeedTreeModel, anyhow::Error> {
    let t: SpeedTreeRaw = quick_xml::de::from_str(contents)?;

    let mut object_names = HashMap::new();
    let mut merged_objects = HashMap::new();
    for object in &t.objects.objects {
        if let Some(parent_id) = object.parent_id {
            let points = object.points.as_ref().unwrap();
            let vertices = object.vertices.as_ref().unwrap();
            let triangles = object.triangles.as_ref().unwrap();

            println!("{} {}", parent_id, object.name);
            println!("{:?}\n{:?}", &triangles.point_indices.0[..10], &triangles.vertex_indices.0[..10]);
            let (output_vertices, output_indices) =
                merged_objects.entry(parent_id).or_insert((Vec::new(), Vec::new()));

            assert_eq!(triangles.count * 3, triangles.point_indices.len() as u32);

            let mut ids_to_index = HashMap::new();
            for tri_index in 0..(triangles.count * 3) as usize {
                let pid = triangles.point_indices[tri_index] as usize;
                let vid = triangles.vertex_indices[tri_index] as usize;

                if let Some(index) = ids_to_index.get(&(pid, vid)) {
                    output_indices.push(*index);
                    continue;
                }
                let index = output_vertices.len() as u32;
                ids_to_index.insert((pid, vid), index);
                output_indices.push(index);
                output_vertices.push(Vertex {
                    position: [points.x[pid], points.y[pid], points.z[pid]],
                    lod_position: [points.lod_x[pid], points.lod_y[pid], points.lod_z[pid]],
                    normal: [
                        vertices.normal_x[vid],
                        vertices.normal_y[vid],
                        vertices.normal_z[vid],
                    ],
                    binormal: [
                        vertices.binormal_x[vid],
                        vertices.binormal_y[vid],
                        vertices.binormal_z[vid],
                    ],
                    texcoord_u: vertices.texcoord_u[vid],
                    texcoord_v: vertices.texcoord_v[vid],
                    ao: vertices.ao[vid],
                    color: ((vertices.vertex_color_r[vid].min(1.0).max(0.0) * 255.0) as u32)
                        | ((vertices.vertex_color_g[vid].min(1.0).max(0.0) * 255.0) as u32) << 8
                        | ((vertices.vertex_color_b[vid].min(1.0).max(0.0) * 255.0) as u32) << 16,
                })
            }
        } else {
            object_names.insert(object.name.clone(), object.id);
        }
    }

    let mut lods = Vec::new();
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for name in ["LOD0"/* , "LOD1", "LOD2", "LOD3"*/] {
        if let Some(id) = object_names.get(name) {
            let mut merged = merged_objects.remove(id).unwrap();

            let start = indices.len() as u32;
            let end = (indices.len() + merged.1.len()) as u32;
            for index in merged.1 {
                indices.push(index + start);
            }
            vertices.append(&mut merged.0);
            lods.push(start..end);
        }
    }

    Ok(SpeedTreeModel { vertices, indices, lods })
}

#[cfg(test)]
mod tests {
    #[test]
    fn vertex_size() {
        assert_eq!(std::mem::size_of::<super::Vertex>(), 64);
    }
    #[test]
    fn parse_xml() {
        let t: super::SpeedTreeRaw =
            quick_xml::de::from_str(include_str!("../assets/Tree/Oak_English_Sapling.xml"))
                .unwrap();
    }
}
