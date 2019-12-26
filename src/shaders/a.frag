#line 2

layout(early_fragment_tests) in;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 1) uniform sampler linear;
layout(set = 0, binding = 2) uniform texture2DArray heights;
layout(set = 0, binding = 3) uniform texture2DArray normals;

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texcoord;
layout(location = 2) in vec2 parent_texcoord;
layout(location = 3) in vec2 colors_layer;
layout(location = 4) in vec2 normals_layer;
layout(location = 5) in vec2 splats_layer;
layout(location = 6) in float morph;

void main() {
	vec3 light_direction = normalize(vec3(0.4, 0.7,0.2));
	vec3 normal = normalize(texture(sampler2DArray(normals, linear),
									vec3(fract(texcoord), normals_layer.x)).xyz);

	float nDotL = dot(normal, light_direction);
	out_color = vec4(vec3(nDotL) * 0.7, 1.0f);

	// Gamma correct output color
	out_color.rgb = pow(out_color.rgb, vec3(2.2));
}
