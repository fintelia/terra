#extension GL_EXT_samplerless_texture_functions: require

struct Globals {
    mat4 view_proj;
	mat4 view_proj_inverse;
	vec4 frustum_planes[5];
	vec3 camera;
	vec3 sun_direction;
};

struct LayerDesc {
	vec3 origin;
	float _step;
	vec3 parent_origin;
	float parent_step;
};
struct NodeState {
    LayerDesc displacements;
	LayerDesc albedo;
	LayerDesc roughness;
	LayerDesc normals;
	vec3 grass_canopy_origin;
	float grass_canopy_step;
	uint resolution;
	uint face;
	uint level;
	uint node_index;
	vec3 relative_position;
	float min_distance;
	vec3 parent_relative_position;
	uint slot;
	uvec2 base_origin;
	uvec2 padding2;
	vec4 padding3[3];
};

struct Indirect {
    uint vertex_count;
    uint instance_count;
    uint base_index;
    uint vertex_offset;
    uint base_instance;
};

struct NodeSlot {
	vec2 layer_origins[16];
	float layer_steps[16];
	int layer_slots[16];

	vec3 relative_position;
	float min_distance;

	uint face;
	uint level;

	uint padding2[58];
};

float extract_height(uint encoded) {
	return (float(encoded & 0x7fffff) * (1 / 512.0)) - 1024.0;
}
float extract_height_above_water(uint encoded) {
	float height = extract_height(encoded);
	if ((encoded & 0x800000) != 0) {
		height = max(height, 0);
	}
	return height;
}

const uint NUM_LAYERS = 8;

const uint DISPLACEMENTS_LAYER = 0;
const uint ALBEDO_LAYER = 1;
const uint ROUGHNESS_LAYER = 2;
const uint NORMALS_LAYER = 3;
const uint HEIGHTMAPS_LAYER = 4;
const uint GRASS_CANOPY_LAYER = 5;
const uint MATERIAL_KIND_LAYER = 6;

const uint PARENT_DISPLACEMENTS_LAYER = NUM_LAYERS + 0;
const uint PARENT_ALBEDO_LAYER = NUM_LAYERS + 1;
const uint PARENT_ROUGHNESS_LAYER = NUM_LAYERS + 2;
const uint PARENT_NORMALS_LAYER = NUM_LAYERS + 3;
const uint PARENT_HEIGHTMAPS_LAYER = NUM_LAYERS + 4;
const uint PARENT_GRASS_CANOPY_LAYER = NUM_LAYERS + 5;
const uint PARENT_MATERIAL_KIND_LAYER = NUM_LAYERS + 6;