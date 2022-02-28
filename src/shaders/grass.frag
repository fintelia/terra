#version 450 core
#include "declarations.glsl"
#include "pbr.glsl"

layout(early_fragment_tests) in;

layout(set = 0, binding = 0) uniform UniformBlock {
	Globals globals;
};
layout(set = 0, binding = 8, std430) readonly buffer NodeBlock {
	Node nodes[];
};

// layout(set = 0, binding = 1, std140) uniform NodeBlock {
// 	vec3 relative_position;
// 	float min_distance;
// 	vec3 parent_relative_position;
// 	float padding1;

//     uint slot;
//     uvec3 padding2;
// } node;

// layout(set = 0, binding = 3) uniform sampler linear;
// layout(set = 0, binding = 4) uniform texture2DArray normals;
// layout(set = 0, binding = 5) uniform texture2DArray albedo;
// layout(set = 0, binding = 6) uniform texture2DArray roughness;


layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec2 texcoord;
layout(location = 3) in vec3 normal;
// layout(location = 4) flat in uint instance;

layout(location = 0) out vec4 out_color;

vec3 extract_normal(vec2 n) {
	n = n * 2.0 - vec2(1.0);
	float y = sqrt(max(1.0 - dot(n, n),0));
	return normalize(vec3(n.x, y, n.y));
}

void main() {
    out_color = vec4(color, 1);

    // vec3 albedo_value = texture(sampler2DArray(albedo, linear), vec3(texcoord, node.nodes_slot)).xyz;
    // vec3 snormal = extract_normal(texture(sampler2DArray(normals, linear), layer_to_texcoord(NORMALS_LAYER)).xy);
	float roughness_value = 0.5;

	out_color = vec4(1);
	out_color.rgb = pbr(color,
						roughness_value,
						position,
						normal,
						globals.camera,
						globals.sun_direction,
						vec3(100000.0));

	out_color.rgb += pbr(color,
						roughness_value,
						position,
						-normal,
						globals.camera,
						globals.sun_direction,
						vec3(100000.0));

	// out_color.rgb = out_color.rgb * 0.3 + 0.7 * pbr(color,
	// 					roughness_value,
	// 					position,
	// 					snormal,
	// 					globals.camera,
	// 					normalize(vec3(0.4, .7, 0.2)),
	// 					vec3(100000.0));

   	float ev100 = 15.0;
	float exposure = 1.0 / (pow(2.0, ev100) * 1.2);
	out_color = tonemap(out_color, exposure, 2.2);
}