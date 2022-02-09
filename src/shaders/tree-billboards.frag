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

layout(set = 0, binding = 1) uniform texture2DArray aerial_perspective;
layout(set = 0, binding = 3) uniform sampler linear;

layout(binding = 4) uniform texture2DArray billboards_albedo;
layout(binding = 5) uniform texture2DArray billboards_normals;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec2 texcoord;
layout(location = 3) in vec3 normal;
layout(location = 4) flat in uint slot;
layout(location = 5) in vec3 horizontal;

layout(location = 0) out vec4 out_color;

vec3 extract_normal(vec2 n) {
	n = n * 2.0 - vec2(1.0);
	float y = sqrt(max(1.0 - dot(n, n),0));
	return normalize(vec3(n.x, y, n.y));
}
vec3 layer_to_texcoord(uint layer) {
	Node node = nodes[slot];
	return vec3(node.layer_origins[layer] + texcoord * node.layer_steps[layer], node.layer_slots[layer]);
}

void main() {
	vec4 albedo = texture(sampler2DArray(billboards_albedo, linear), vec3(texcoord/6.0+1./6, 0));
	vec2 tx_normal = texture(sampler2DArray(billboards_normals, linear), vec3(texcoord/6.0+1./6, 0)).xy;

	vec3 true_normal = normalize(tx_normal.y * horizontal - tx_normal.x * vec3(0,1,0) - normal * sqrt(1-dot(tx_normal, tx_normal)));

	out_color = vec4(1);
	out_color.rgb = pbr(albedo.rgb,
						0.5,
						position,
						vec3(0,1,0),//true_normal,
						globals.camera,
						normalize(vec3(0.4, .7, 0.2)),
						vec3(100000.0));

	vec4 ap = texture(sampler2DArray(aerial_perspective, linear), layer_to_texcoord(AERIAL_PERSPECTIVE_LAYER));
	out_color.rgb *= ap.a * 16.0;
	out_color.rgb += ap.rgb * 16.0;

	if (albedo.a < 1)
		discard;


   	float ev100 = 15.0;
	float exposure = 1.0 / (pow(2.0, ev100) * 1.2);
	out_color = tonemap(out_color, exposure, 2.2);
}
