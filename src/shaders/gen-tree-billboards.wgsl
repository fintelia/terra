struct Entry {
    position: vec3<f32>,
    albedo: u32,
    angle: f32,
    height: f32,
    uv: u32,
    padding: f32,
};
struct Entries {
    entries: array<array<Entry, 16384>>,
};

@group(0) @binding(0) var<uniform> ubo: GenMeshUniforms;
@group(0) @binding(1) var<storage, read_write> tree_billboards_storage: Entries;
@group(0) @binding(2) var<storage, read> nodes: Nodes;
@group(0) @binding(3) var<storage, read_write> mesh_indirect: Indirects;
@group(0) @binding(4) var linearsamp: sampler;
@group(0) @binding(5) var nearest: sampler;
@group(0) @binding(6) var displacements: texture_2d_array<f32>;
@group(0) @binding(7) var tree_attributes: texture_2d_array<f32>;


@compute
@workgroup_size(8,8)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let node = nodes.entries[ubo.slot];

    let index = global_id.xy % vec2<u32>(128u);
    let entry = 4u * (global_id.y / 128u) + (global_id.x / 128u);

    let rnd1 = random3(vec3<f32>(vec2<f32>(index), 1.0));
    let rnd2 = random3(vec3<f32>(vec2<f32>(index), 2.0));
    let rnd3 = random3(vec3<f32>(vec2<f32>(index), 3.0));
    let rnd4 = random3(vec3<f32>(vec2<f32>(index), 4.0));
    let rnd5 = random3(vec3<f32>(vec2<f32>(index), 5.0));

    let tree_attr = textureSampleLevel(
        tree_attributes,
        nearest,
        layer_texcoord(node.layers[TREE_ATTRIBUTES_LAYER], vec2<f32>(global_id.xy) / 512.0),
        node.layers[TREE_ATTRIBUTES_LAYER].slot,
        0.0
    );

    if (tree_attr.a == 0.0) {
        return;
    }

    // Sample displacements texture at random offset (rnd1, rnd).
    let uv = (vec2<f32>(global_id.xy) + vec2<f32>(rnd1, rnd2)) / 512.0;
    let array_index = node.layers[DISPLACEMENTS_LAYER].slot;
    let dimensions = textureDimensions(displacements);
    let stexcoord = max(layer_texcoord(node.layers[DISPLACEMENTS_LAYER], uv) * vec2<f32>(dimensions) - vec2<f32>(0.5), vec2<f32>(0.0));
    let f = fract(stexcoord);
    let base_coords = vec2<i32>(stexcoord - f);
    let i00 = textureLoad(displacements, base_coords, array_index, 0);
    let i10 = textureLoad(displacements, min(base_coords + vec2<i32>(1,0), dimensions-vec2<i32>(1)), array_index, 0);
    let i01 = textureLoad(displacements, min(base_coords + vec2<i32>(0,1), dimensions-vec2<i32>(1)), array_index, 0);
    let i11 = textureLoad(displacements, min(base_coords + vec2<i32>(1,1), dimensions-vec2<i32>(1)), array_index, 0);
    let position = mix(mix(i00, i10, f.x), mix(i01, i11, f.x), f.y);

    let i = atomicAdd(&mesh_indirect.entries[ubo.mesh_base_entry + entry].vertex_count, 6) / 6;
    tree_billboards_storage.entries[ubo.storage_base_entry + entry][i].position = position.xyz;
    tree_billboards_storage.entries[ubo.storage_base_entry + entry][i].albedo = pack4x8unorm(vec4<f32>(rnd3, rnd4, rnd5, 1.0));
    tree_billboards_storage.entries[ubo.storage_base_entry + entry][i].angle = 0.0;
    tree_billboards_storage.entries[ubo.storage_base_entry + entry][i].height = 10.0;
    tree_billboards_storage.entries[ubo.storage_base_entry + entry][i].uv = pack2x16unorm(uv);
}
