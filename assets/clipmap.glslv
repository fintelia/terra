#version 450 core

uniform vec3 position;
uniform vec3 scale;
uniform sampler2D heights;
uniform mat4 modelViewProjection;

in vec2 vPosition;

out vec2 texCoord;

void main() {
  texCoord = vPosition;

  float y = texture(heights, vPosition).r;
  vec3 pos = vec3(vPosition.x, y, vPosition.y) * scale + position;
  gl_Position = modelViewProjection * vec4(pos, 1);
}
