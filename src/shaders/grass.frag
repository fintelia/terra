#line 2

layout(early_fragment_tests) in;

layout(set = 0, binding = 0) uniform UniformBlock {
    mat4 view_proj;
	vec3 camera;
	float padding;
} ubo;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

layout(location = 0) out vec4 out_color;

void main() {
    out_color = vec4(color, 1);
}