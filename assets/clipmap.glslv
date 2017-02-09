#version 450 core

uniform int resolution;

out vec2 vPosition;

void main() {
  uint quad = gl_VertexID / 4;
  uint v = gl_VertexID % 4;
  float x = (quad % resolution + (v % 2)) / float(resolution);
  float y = (quad / resolution + (v / 2)) / float(resolution);

  vPosition = vec2(x, y);
}
