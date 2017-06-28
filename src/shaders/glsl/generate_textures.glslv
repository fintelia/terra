#version 450 core

out vec2 vTexCoord;

void main() {
  if(gl_VertexID == 0) gl_Position = vec4(-1, -1, 0, 1);
  if(gl_VertexID == 1) gl_Position = vec4(-1,  1, 0, 1);
  if(gl_VertexID == 2) gl_Position = vec4( 1, -1, 0, 1);

  if(gl_VertexID == 3) gl_Position = vec4( 1,  1, 0, 1);
  if(gl_VertexID == 4) gl_Position = vec4( 1, -1, 0, 1);
  if(gl_VertexID == 5) gl_Position = vec4(-1,  1, 0, 1);

  vTexCoord = gl_Position.xy * 0.5 + vec2(0.5);
}
