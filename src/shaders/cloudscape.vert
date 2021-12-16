#version 450 core
#include "declarations.glsl"

layout(location = 0) out vec4 position;

void main() {
	if(gl_VertexIndex == 0) position = vec4(-1, -1, 0, 1);
	if(gl_VertexIndex == 1) position = vec4(-1,  3, 0, 1);
	if(gl_VertexIndex == 2) position = vec4( 3, -1, 0, 1);
	gl_Position = position;
}
