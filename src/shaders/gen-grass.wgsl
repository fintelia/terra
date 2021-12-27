struct Entry {
    position: vec3<f32>;
    angle: f32;
    albedo: vec3<f32>;
    slant: f32;
    texcoord: vec2<f32>;
    padding1: vec2<f32>;
    padding2: vec4<f32>;
};
struct Node {
    layer_origins: array<vec2<f32>, 16>;
    layer_steps: array<f32, 16>;
    layer_slots: array<i32, 16>;
	relative_position: vec3<f32>;
	min_distance: f32;
	mesh_valid_mask: array<u32, 4>;
    face: u32;
	level: u32;
	padding2: array<u32, 54>;
};
struct Indirect {
    vertex_count: atomic<i32>; // TODO: why doesn't u32 work here?
    instance_count: u32;
    base_index: u32;
    vertex_offset: u32;
    base_instance: u32;
};

struct GenMeshUniforms {
    slot: u32;
    storage_base_entry: u32;
    mesh_base_entry: u32;
    entries_per_node: u32;
};
struct Entries {
    entries: array<array<Entry, 1024>>;
};
struct Indirects {
    entries: array<Indirect>;
};
struct Nodes {
    entries: array<Node>;
};

[[group(0), binding(0)]] var<uniform> ubo: GenMeshUniforms;
[[group(0), binding(1)]] var<storage, read_write> grass_storage: Entries;
[[group(0), binding(2)]] var<storage, read> nodes: Nodes;
[[group(0), binding(3)]] var<storage, read_write> mesh_indirect: Indirects;
[[group(0), binding(4)]] var linear: sampler;
[[group(0), binding(5)]] var displacements: texture_2d_array<f32>;
[[group(0), binding(6)]] var normals: texture_2d_array<f32>;
[[group(0), binding(7)]] var albedo: texture_2d_array<f32>;
[[group(0), binding(8)]] var grass_canopy: texture_2d_array<f32>;

let NUM_LAYERS: u32 = 8u;

let DISPLACEMENTS_LAYER: u32 = 0u;
let ALBEDO_LAYER: u32 = 1u;
let ROUGHNESS_LAYER: u32 = 2u;
let NORMALS_LAYER: u32 = 3u;
let HEIGHTMAPS_LAYER: u32 = 4u;
let GRASS_CANOPY_LAYER: u32 = 5u;
let MATERIAL_KIND_LAYER: u32 = 6u;
let AERIAL_PERSPECTIVE_LAYER: u32 = 7u;

let PARENT_DISPLACEMENTS_LAYER: u32 = 8u;
let PARENT_ALBEDO_LAYER: u32 = 9u;
let PARENT_ROUGHNESS_LAYER: u32 = 10u;
let PARENT_NORMALS_LAYER: u32 = 11u;
let PARENT_HEIGHTMAPS_LAYER: u32 = 12u;
let PARENT_GRASS_CANOPY_LAYER: u32 = 13u;
let PARENT_MATERIAL_KIND_LAYER: u32 = 14u;

let GRASS_BASE_SLOT: u32 = 574u;//30 + (19 - 2) * 32;

fn hash(x: u32) -> u32 {
    var xx = x;
    xx = xx + ( xx << 10u );
    xx = xx ^ ( xx >>  6u );
    xx = xx + ( xx <<  3u );
    xx = xx ^ ( xx >> 11u );
    xx = xx + ( xx << 15u );
    return xx;
}
fn hash2(v: vec2<u32> ) -> u32 { return hash( v.x ^ hash(v.y)                         ); }
fn hash3(v: vec3<u32> ) -> u32 { return hash( v.x ^ hash(v.y) ^ hash(v.z)             ); }
fn hash4(v: vec4<u32> ) -> u32 { return hash( v.x ^ hash(v.y) ^ hash(v.z) ^ hash(v.w) ); }
fn floatConstruct(m: u32) -> f32 { return bitcast<f32>((m & 0x007FFFFFu) | 0x3F800000u) - 1.0; }
fn random(x: f32) -> f32 { return floatConstruct(hash(bitcast<u32>(x))); }
fn random2(x: vec2<f32>) -> f32 { return floatConstruct(hash2(bitcast<vec2<u32>>(x))); }
fn random3(x: vec3<f32>) -> f32 { return floatConstruct(hash3(bitcast<vec3<u32>>(x))); }
fn random4(x: vec4<f32>) -> f32 { return floatConstruct(hash4(bitcast<vec4<u32>>(x))); }

fn extract_normal(n: vec2<f32>) -> vec3<f32> {
    let n = n * 2.0 - vec2<f32>(1.0);
	let y = sqrt(max(1.0 - dot(n, n), 0.0));
	return normalize(vec3<f32>(n.x, y, n.y));
}

// fn layer_to_texcoord(layer: u32, global_id: vec3<u32>) -> vec3<f32> {
// 	var node = nodes.entries[ubo.slot];
//     let texcoord = vec2<f32>(global_id.xy) / 128.0 * 64.0;
// 	return vec3<f32>(node.layer_origins[layer] + texcoord * node.layer_steps[layer], f32(node.layer_slots[layer]));
// }

fn read_texture(layer: u32, global_id: vec3<u32>) -> vec4<f32> {
	var node = nodes.entries[ubo.slot];
    let texcoord = vec2<f32>(global_id.xy) / 128.0 * 64.0;
    let texcoord = node.layer_origins[layer] + texcoord * node.layer_steps[layer];
    let array_index = node.layer_slots[layer];

    let l = layer % NUM_LAYERS;
    if (l == ALBEDO_LAYER) {            return textureSampleLevel(albedo, linear, texcoord, array_index, 0.0); }
    else if (l == NORMALS_LAYER) {           return textureSampleLevel(normals, linear, texcoord, array_index, 0.0); }
    else if (l == GRASS_CANOPY_LAYER) {      return textureSampleLevel(grass_canopy, linear, texcoord, array_index, 0.0); }
    else if (l == DISPLACEMENTS_LAYER) {
        let dimensions = textureDimensions(displacements);
        let f = fract(texcoord.xy * vec2<f32>(dimensions));
        let base_coords = vec2<i32>(texcoord.xy * vec2<f32>(dimensions));
        let i00 = textureLoad(displacements, base_coords, array_index, 0);
        let i10 = textureLoad(displacements, base_coords + vec2<i32>(1,0), array_index, 0);
        let i01 = textureLoad(displacements, base_coords + vec2<i32>(0,1), array_index, 0);
        let i11 = textureLoad(displacements, base_coords + vec2<i32>(1,1), array_index, 0);
        return mix(mix(i00, i10, f.x), mix(i01, i11, f.y), f.y);
    }

    return vec4<f32>(1.0, 0.0, 1.0, 1.0);
}

[[stage(compute), workgroup_size(8,8)]]
fn main(
    [[builtin(global_invocation_id)]] global_id: vec3<u32>,
) {
    let node = nodes.entries[ubo.slot];

    let index = global_id.xy % vec2<u32>(32u);
    let entry = 4u * (global_id.y / 32u) + (global_id.x / 32u);

    if (all(index == vec2<u32>(0u))) {
       mesh_indirect.entries[ubo.mesh_base_entry + entry].instance_count = 1u;
    }

    let rnd1 = random3(vec3<f32>(vec2<f32>(index), 1.0));
    let rnd2 = random3(vec3<f32>(vec2<f32>(index), 2.0));
    let rnd3 = random3(vec3<f32>(vec2<f32>(index), 3.0));
    let rnd4 = random3(vec3<f32>(vec2<f32>(index), 4.0));
    let rnd5 = random3(vec3<f32>(vec2<f32>(index), 5.0));

    let normal = extract_normal(read_texture(NORMALS_LAYER, global_id).xy);
    let albedo_value = read_texture(ALBEDO_LAYER, global_id).xyz;
    let canopy = read_texture(GRASS_CANOPY_LAYER, global_id);

    if (normal.y < 0.95) {
        return;
    }

    let i = atomicAdd(&mesh_indirect.entries[ubo.mesh_base_entry + entry].vertex_count, 15) / 15;
    grass_storage.entries[ubo.storage_base_entry + entry][i].texcoord = vec2<f32>(0.0); //layer_to_texcoord(NORMALS_LAYER).xy;
    grass_storage.entries[ubo.storage_base_entry + entry][i].position = read_texture(DISPLACEMENTS_LAYER, global_id).xyz;
    grass_storage.entries[ubo.storage_base_entry + entry][i].albedo = ((canopy.rgb - 0.5) * 0.025 + albedo_value + vec3<f32>(-.0)) * mix(vec3<f32>(.5), vec3<f32>(1.5), vec3<f32>(rnd2, rnd3, rnd4));
    grass_storage.entries[ubo.storage_base_entry + entry][i].angle = rnd5 * 2.0 * 3.14159265;
    grass_storage.entries[ubo.storage_base_entry + entry][i].slant = rnd1;
}
