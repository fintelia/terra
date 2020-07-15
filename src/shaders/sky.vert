layout(location = 0) out vec2 position;

void main() {
	if(gl_VertexIndex == 0) gl_Position = vec4(-1, -1, 0, 1e-8);
	if(gl_VertexIndex == 1) gl_Position = vec4(-1,  3, 0, 1e-8);
	if(gl_VertexIndex == 2) gl_Position = vec4( 3, -1, 0, 1e-8);
	position = gl_Position.xy;
}
