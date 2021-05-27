#extension GL_EXT_samplerless_texture_functions: require
#extension GL_ARB_compute_shader: require

struct Globals {
    mat4 view_proj;
	mat4 view_proj_inverse;
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
	float padding1;
	vec4 padding2[4];
};
