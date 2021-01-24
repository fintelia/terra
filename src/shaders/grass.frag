#line 2

layout(early_fragment_tests) in;

layout(set = 0, binding = 0) uniform UniformBlock {
    mat4 view_proj;
	vec3 camera;
	float padding;
} ubo;

layout(set = 0, binding = 1, std140) uniform NodeBlock {
	vec3 relative_position;
	float min_distance;
	vec3 parent_relative_position;
	float padding1;

    uint slot;
    uvec3 padding2;
} node;

layout(set = 0, binding = 3) uniform sampler linear;
layout(set = 0, binding = 4) uniform texture2DArray normals;
layout(set = 0, binding = 5) uniform texture2DArray albedo;
layout(set = 0, binding = 6) uniform texture2DArray roughness;


layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec2 texcoord;

layout(location = 0) out vec4 out_color;

vec3 extract_normal(vec2 n) {
	n = n * 2.0 - vec2(1.0);
	float y = sqrt(max(1.0 - dot(n, n),0));
	return normalize(vec3(n.x, y, n.y));
}

void main() {
    out_color = vec4(color, 1);

    // vec3 albedo_value = texture(sampler2DArray(albedo, linear), vec3(texcoord, node.nodes_slot)).xyz;
    // vec3 normal = extract_normal(texture(sampler2DArray(normals, linear), vec3(texcoord, node.nodes_slot)).xy);
	float roughness_value = 0.9;

	out_color = vec4(1);
	out_color.rgb = pbr(color,
						roughness_value,
						position,
						normalize(vec3(1)), // normal
						ubo.camera,
						normalize(vec3(0.4, .7, 0.2)),
						vec3(100000.0)) * .8;

   	float ev100 = 15.0;
	float exposure = 1.0 / (pow(2.0, ev100) * 1.2);
	out_color = tonemap(out_color, exposure, 2.2);
}