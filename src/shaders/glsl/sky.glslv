out vec2 position;

void main() {
	if(gl_VertexID == 0) gl_Position = vec4(-1, -1, 1, 1);
	if(gl_VertexID == 1) gl_Position = vec4(-1,  1, 1, 1);
	if(gl_VertexID == 2) gl_Position = vec4( 1,  1, 1, 1);

	if(gl_VertexID == 3) gl_Position = vec4(-1, -1, 1, 1);
	if(gl_VertexID == 5) gl_Position = vec4( 1,  1, 1, 1);
	if(gl_VertexID == 4) gl_Position = vec4( 1, -1, 1, 1);
	position = gl_Position.xy;
}
