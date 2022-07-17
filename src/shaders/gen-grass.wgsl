struct Entry {
    position: vec3<f32>,
    angle: f32,
    albedo: vec3<f32>,
    slant: f32,
    texcoord: vec2<f32>,
    padding1: vec2<f32>,
    padding2: vec4<f32>,
};
struct Entries {
    entries: array<array<Entry, 1024>>,
};

@group(0) @binding(0) var<uniform> ubo: GenMeshUniforms;
@group(0) @binding(1) var<storage, read_write> grass_storage: Entries;
@group(0) @binding(3) var<storage, read_write> mesh_indirect: Indirects;
@group(0) @binding(4) var<storage, read> nodes: Nodes;
@group(0) @binding(5) var linearsamp: sampler;
@group(0) @binding(6) var displacements: texture_2d_array<f32>;
@group(0) @binding(7) var normals: texture_2d_array<f32>;
@group(0) @binding(8) var albedo: texture_2d_array<f32>;
@group(0) @binding(9) var grass_canopy: texture_2d_array<f32>;

fn read_texture(layer: u32, global_id: vec3<u32>) -> vec4<f32> {
	var node = nodes.entries[ubo.slot];
    let texcoord = vec2<f32>(global_id.xy) / 128.0;
    let texcoord = node.layer_origins[layer] + texcoord * node.layer_ratios[layer];
    let array_index = node.layer_slots[layer];

    let l = layer % NUM_LAYERS;
    if (l == ALBEDO_LAYER) {            return textureSampleLevel(albedo, linearsamp, texcoord, array_index, 0.0); }
    else if (l == NORMALS_LAYER) {           return textureSampleLevel(normals, linearsamp, texcoord, array_index, 0.0); }
    else if (l == GRASS_CANOPY_LAYER) {      return textureSampleLevel(grass_canopy, linearsamp, texcoord, array_index, 0.0); }
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

@compute
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

    // let texcoord = vec2<f32>(global_id.xy) / 128.0;
    let normal = extract_normal(read_texture(NORMALS_LAYER, global_id).xy);
    let albedo_value = read_texture(ALBEDO_LAYER, global_id).xyz;
    let canopy = read_texture(GRASS_CANOPY_LAYER, global_id);

    if (normal.y < 0.95) {
        return;
    }

    // Sample displacements texture at random offset (rnd1, rnd).
    let texcoord = (vec2<f32>(global_id.xy) + vec2<f32>(rnd1, rnd2)) / 128.0;
    let stexcoord = node.layer_origins[DISPLACEMENTS_LAYER] + texcoord * node.layer_ratios[DISPLACEMENTS_LAYER];
    let array_index = node.layer_slots[DISPLACEMENTS_LAYER];
    let dimensions = textureDimensions(displacements);
    let stexcoord = max(stexcoord.xy * vec2<f32>(dimensions) - vec2<f32>(0.5), vec2<f32>(0.0));
    let f = fract(stexcoord);
    let base_coords = vec2<i32>(stexcoord - f);
    let i00 = textureLoad(displacements, base_coords, array_index, 0);
    let i10 = textureLoad(displacements, min(base_coords + vec2<i32>(1,0), dimensions-vec2<i32>(1)), array_index, 0);
    let i01 = textureLoad(displacements, min(base_coords + vec2<i32>(0,1), dimensions-vec2<i32>(1)), array_index, 0);
    let i11 = textureLoad(displacements, min(base_coords + vec2<i32>(1,1), dimensions-vec2<i32>(1)), array_index, 0);
    let position = mix(mix(i00, i10, f.x), mix(i01, i11, f.x), f.y);

    let i = atomicAdd(&mesh_indirect.entries[ubo.mesh_base_entry + entry].vertex_count, 15) / 15;
    grass_storage.entries[ubo.storage_base_entry + entry][i].texcoord = texcoord; //layer_to_texcoord(NORMALS_LAYER).xy;
    grass_storage.entries[ubo.storage_base_entry + entry][i].position = position.xyz;
    grass_storage.entries[ubo.storage_base_entry + entry][i].albedo = ((canopy.rgb - 0.5) * 0.025 + albedo_value) * mix(vec3<f32>(.75), vec3<f32>(1.25), vec3<f32>(rnd2, rnd3, rnd4));
    grass_storage.entries[ubo.storage_base_entry + entry][i].angle = rnd5 * 2.0 * 3.14159265;
    grass_storage.entries[ubo.storage_base_entry + entry][i].slant = rnd1;
}
