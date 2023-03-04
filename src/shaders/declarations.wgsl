
struct Node {
    node_center: array<vec2<u32>, 4>,

    layer_origins: array<vec2<f32>, 48>,
    layer_ratios: array<f32, 48>,
    layer_slots: array<i32, 48>,
	relative_position: vec3<f32>,
	min_distance: f32,
	mesh_valid_mask: array<u32, 4>,
    face: u32,
	level: u32,
    coords: vec2<u32>,

    parent: i32,
	padding2: array<u32, 43>,
};
struct Nodes {
    entries: array<Node>,
};

struct GenMeshUniforms {
    slot: u32,
    storage_base_entry: u32,
    mesh_base_entry: u32,
    entries_per_node: u32,
};

struct Indirect {
    vertex_count: atomic<i32>, // TODO: why doesn't u32 work here?
    instance_count: u32,
    base_index: u32,
    vertex_offset: u32,
    base_instance: u32,
};
struct Indirects {
    entries: array<Indirect>,
};

let NUM_LAYERS: u32 = 24u;

let BASE_HEIGHTMAPS_LAYER: u32 = 0u;
let DISPLACEMENTS_LAYER: u32 = 1u;
let ALBEDO_LAYER: u32 = 2u;
let NORMALS_LAYER: u32 = 3u;
let GRASS_CANOPY_LAYER: u32 = 4u;
let TREE_ATTRIBUTES_LAYER: u32 = 5u;
let AERIAL_PERSPECTIVE_LAYER: u32 = 6u;
let BENT_NORMALS_LAYER: u32 = 7u;

let PARENT_HEIGHTMAPS_LAYER: u32 = 24u;
let PARENT_DISPLACEMENTS_LAYER: u32 = 25u;
let PARENT_ALBEDO_LAYER: u32 = 26u;
let PARENT_NORMALS_LAYER: u32 = 27u;
let PARENT_GRASS_CANOPY_LAYER: u32 = 28u;
let PARENT_TREE_ATTRIBUTES_LAYER: u32 = 30u;

let GRASS_BASE_SLOT: u32 = 540u;//30 + (19 - 2) * 30;

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
