#version 450 core

out vec3 vPosition;

void main() {
  if(gl_VertexID == 0) vPosition = vec3(-3, 0, 3);
  if(gl_VertexID == 1) vPosition = vec3(3, 0, 3);
  if(gl_VertexID == 2) vPosition = vec3(-3, 0, -3);
  if(gl_VertexID == 3) vPosition = vec3(3, 0, -3);
}
