struct Entry {
    position: vec3<f32>;
    angle: f32;
    albedo: vec3<f32>;
    height: f32;
    padding0: vec4<f32>;
    padding1: vec4<f32>;
};
struct Entries {
    entries: array<array<Entry, 1024>>;
};

@group(0) @binding(0) var<uniform> ubo: GenMeshUniforms;
@group(0) @binding(1) var<storage, read_write> tree_billboards_storage: Entries;
@group(0) @binding(2) var<storage, read> nodes: Nodes;
@group(0) @binding(3) var<storage, read_write> mesh_indirect: Indirects;
@group(0) @binding(4) var linear: sampler;
@group(0) @binding(5) var nearest: sampler;
@group(0) @binding(6) var displacements: texture_2d_array<f32>;
@group(0) @binding(7) var tree_attributes: texture_2d_array<f32>;


@stage(compute)
@workgroup_size(8,8)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let node = nodes.entries[ubo.slot];

    let index = global_id.xy % vec2<u32>(32u);
    let entry = 4u * (global_id.y / 32u) + (global_id.x / 32u);

    let rnd1 = random3(vec3<f32>(vec2<f32>(index), 1.0));
    let rnd2 = random3(vec3<f32>(vec2<f32>(index), 2.0));
    let rnd3 = random3(vec3<f32>(vec2<f32>(index), 3.0));
    let rnd4 = random3(vec3<f32>(vec2<f32>(index), 4.0));
    let rnd5 = random3(vec3<f32>(vec2<f32>(index), 5.0));

    let texcoord = vec2<f32>(global_id.xy) / 128.0;
    let texcoord = node.layer_origins[TREE_ATTRIBUTES_LAYER] + texcoord * node.layer_ratios[TREE_ATTRIBUTES_LAYER];
    let array_index = node.layer_slots[TREE_ATTRIBUTES_LAYER];
    let tree_attr = textureSampleLevel(tree_attributes, nearest, texcoord, array_index, 0.0);

    if (tree_attr.a == 0.0) {
        return;
    }

    // Sample displacements texture at random offset (rnd1, rnd).
    let texcoord = (vec2<f32>(global_id.xy) + vec2<f32>(rnd1, rnd2)) / 128.0;
    let texcoord = node.layer_origins[DISPLACEMENTS_LAYER] + texcoord * node.layer_ratios[DISPLACEMENTS_LAYER];
    let array_index = node.layer_slots[DISPLACEMENTS_LAYER];
    let dimensions = textureDimensions(displacements);
    let f = fract(texcoord.xy * vec2<f32>(dimensions));
    let base_coords = vec2<i32>(texcoord.xy * vec2<f32>(dimensions));
    let i00 = textureLoad(displacements, base_coords, array_index, 0);
    let i10 = textureLoad(displacements, base_coords + vec2<i32>(1,0), array_index, 0);
    let i01 = textureLoad(displacements, base_coords + vec2<i32>(0,1), array_index, 0);
    let i11 = textureLoad(displacements, base_coords + vec2<i32>(1,1), array_index, 0);
    let position = mix(mix(i00, i10, f.x), mix(i01, i11, f.y), f.y);

    let i = atomicAdd(&mesh_indirect.entries[ubo.mesh_base_entry + entry].vertex_count, 6) / 6;
    tree_billboards_storage.entries[ubo.storage_base_entry + entry][i].position = position.xyz;
    tree_billboards_storage.entries[ubo.storage_base_entry + entry][i].albedo = vec3<f32>(rnd3, rnd4, rnd5);
    tree_billboards_storage.entries[ubo.storage_base_entry + entry][i].angle = 0.0;
    tree_billboards_storage.entries[ubo.storage_base_entry + entry][i].height = 10.0;
}
