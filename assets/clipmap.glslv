#version 450 core

uniform ivec2 flipAxis;

uniform int resolution;
uniform vec3 position;
uniform vec3 scale;
uniform sampler2D heights;
uniform mat4 modelViewProjection;

in ivec2 vPosition;

out vec2 texCoord;

void main() {
  vec2 iPosition = mix(vPosition, ivec2(resolution-1) - vPosition, flipAxis);
  texCoord = iPosition / vec2(resolution + 1);

  float y = texture(heights, texCoord).r * 0.0;
  vec2 p = iPosition / vec2(resolution - 1);
  vec3 pos = vec3(p.x, y, p.y) * scale + position;
  gl_Position = modelViewProjection * vec4(pos, 1);
}
