#extension GL_EXT_samplerless_texture_functions: require

#ifndef xdouble
#define xdouble uvec2
#endif

struct Globals {
    mat4 view_proj;
	mat4 view_proj_inverse;
	vec4 frustum_planes[5];
	vec3 camera;
	vec3 sun_direction;
};

struct Indirect {
    uint vertex_count;
    uint instance_count;
    uint base_index;
    uint vertex_offset;
    uint base_instance;
};

struct Node {
    xdouble node_center_x;
    xdouble node_center_y;
    xdouble node_center_z;
    xdouble padding0;

	vec2 layer_origins[48];
	float layer_steps[48];
	int layer_slots[48];

	vec3 relative_position;
	float min_distance;

	uint mesh_valid_mask[4];

	uint face;
	uint level;
	uvec2 coords;

	int parent;
	uint padding[43];
};

struct GenMeshUniforms {
	uint slot;
    uint storage_base_entry;
    uint mesh_base_entry;
    uint entries_per_node;
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

const uint NUM_LAYERS = 24;

const uint HEIGHTMAPS_LAYER = 0;
const uint DISPLACEMENTS_LAYER = 1;
const uint ALBEDO_LAYER = 2;
const uint NORMALS_LAYER = 3;
const uint GRASS_CANOPY_LAYER = 4;
const uint TREE_ATTRIBUTES_LAYER = 5;
const uint AERIAL_PERSPECTIVE_LAYER = 6;
const uint BENT_NORMALS_LAYER = 7;
const uint TREECOVER_LAYER = 8;
const uint BASE_ALBEDO_LAYER = 9;

const uint PARENT_HEIGHTMAPS_LAYER = NUM_LAYERS + HEIGHTMAPS_LAYER;
const uint PARENT_DISPLACEMENTS_LAYER = NUM_LAYERS + DISPLACEMENTS_LAYER;
const uint PARENT_ALBEDO_LAYER = NUM_LAYERS + ALBEDO_LAYER;
const uint PARENT_NORMALS_LAYER = NUM_LAYERS + NORMALS_LAYER;
const uint PARENT_GRASS_CANOPY_LAYER = NUM_LAYERS + GRASS_CANOPY_LAYER;
const uint PARENT_TREE_ATTRIBUTES_LAYER = NUM_LAYERS + TREE_ATTRIBUTES_LAYER;
const uint PARENT_TREECOVER_LAYER = 8;

const uint TREE_ATTRIBUTES_BASE_SLOT = 30 + (11 - 2) * 32;

const uint GRASS_BASE_SLOT = 30 + (19 - 2) * 32;
const uint TREE_BILLBOARDS_BASE_SLOT = 30 + (14 - 2) * 32;
