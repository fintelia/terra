#line 2
uniform ivec2 flipAxis;
uniform int resolution;
uniform mat4 modelViewProjection;
uniform vec3 position;
uniform vec3 scale;

uniform vec2 textureOffset;
uniform float textureStep;

in uvec2 vPosition;
out vec2 texCoord;
out vec3 fPosition;

void main() {
  vec2 iPosition = mix(ivec2(vPosition), ivec2(resolution-1) - ivec2(vPosition), flipAxis);

  vec2 tPosition = textureOffset + iPosition * textureStep;
  texCoord = (vec2(tPosition) + vec2(0.5)) / textureSize(heights, 0);

  vec2 p = iPosition / vec2(resolution - 1);
  fPosition = vec3(p.x, 0, p.y) * scale + position;

  float y;
  vec2 slope;
  compute_height_and_slope(fPosition.xz, texCoord, y, slope);

  fPosition = vec3(p.x, y, p.y) * scale + position;
  gl_Position = modelViewProjection * vec4(fPosition, 1.0);
}
