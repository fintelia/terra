#line 2

layout(set = 0, binding = 0) uniform UniformBlock {
    mat4 view_proj;
	dvec3 camera;
	double padding;
} ubo;

layout(location = 0) in vec2 position;
layout(location = 0) out vec4 OutColor;

void main() {
    OutColor = vec4(0, 0.3, 8, 1);
}
