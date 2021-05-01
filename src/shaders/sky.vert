
layout(set = 0, binding = 0) uniform UniformBlock {
    mat4 view_proj;
	mat4 view_proj_inverse;
	vec3 camera;
	float padding;
} ubo;

layout(location = 0) out vec4 position;

void main() {
	if(gl_VertexIndex == 0) position = vec4(-1, -1, 0, 1);
	if(gl_VertexIndex == 1) position = vec4(-1,  3, 0, 1);
	if(gl_VertexIndex == 2) position = vec4( 3, -1, 0, 1);
	gl_Position = position;
}
