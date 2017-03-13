#version 450 core

uniform int resolution;
uniform vec3 position;
uniform vec3 scale;
uniform sampler2D heights;
uniform mat4 modelViewProjection;

in ivec2 vPosition;

out vec2 texCoord;

void main() {
  texCoord = vPosition / vec2(resolution + 1);

  float y = texture(heights, texCoord).r * 0.1;
  vec2 p = vPosition / vec2(resolution + 1);
  vec3 pos = vec3(p.x, y, p.y) * scale + position;
  gl_Position = modelViewProjection * vec4(pos, 1);
}
