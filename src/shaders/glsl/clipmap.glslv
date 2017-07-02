#version 450 core

uniform ivec2 flipAxis;
uniform int resolution;
uniform vec3 position;
uniform vec3 scale;
uniform mat4 modelViewProjection;

uniform sampler2D heights;
uniform ivec2 textureOffset;
uniform int textureStep;

in ivec2 vPosition;
out vec2 rawTexCoord;
out vec2 texCoord;

void main() {
  vec2 iPosition = mix(vPosition, ivec2(resolution-1) - vPosition, flipAxis);

  vec2 tPosition = textureOffset + iPosition * textureStep;
  rawTexCoord = textureOffset + iPosition * textureStep;
  texCoord = (vec2(tPosition) + vec2(0.5)) / textureSize(heights, 0);
  float y = texture(heights, texCoord).r;

  vec2 p = iPosition / vec2(resolution - 1);
  vec3 pos = vec3(p.x, y, p.y) * scale + position;
  gl_Position = modelViewProjection * vec4(pos, 1.0);
}
