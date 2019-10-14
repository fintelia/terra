#line 2

layout(binding = 0) uniform UniformBlock {
    mat4 view;
    mat4 projection;
	vec3 camera;
	float padding;
} uniform_block;

void main() {
	vec4 position = vec4(0);
	if(gl_VertexIndex == 0) position = vec4(0, 0, 0, 1);
	if(gl_VertexIndex == 1) position = vec4(-10, 0, -10, 1);
	if(gl_VertexIndex == 2) position = vec4(10, 0, -10, 1);

	gl_Position = uniform_block.projection * uniform_block.view * position;
}
