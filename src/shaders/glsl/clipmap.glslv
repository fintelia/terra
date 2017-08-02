#version 330 core

uniform ivec2 flipAxis;
uniform int resolution;
uniform vec3 position;
uniform vec3 scale;
uniform mat4 modelViewProjection;

uniform sampler2D heights;
uniform sampler2D slopes;
uniform sampler2D detail;

uniform vec2 textureOffset;
uniform float textureStep;
uniform int vertexFractalOctaves;

in uvec2 vPosition;
out vec2 rawTexCoord;
out vec2 texCoord;

/// Uses a fractal to refine the height and slope sourced from the course texture.
void compute_height_and_slope(inout float height, inout vec2 slope) {
	vec2 invTextureSize = 1.0  / textureSize(detail,0);

	float smoothing = mix(0.1, 1.0, smoothstep(0.0, 1.0, length(slope)));

	float scale = 10.0;
	float texCoordScale = 8.0;
	for(int i = 0; i < vertexFractalOctaves; i++) {
		vec3 v = texture(detail, (rawTexCoord * texCoordScale + 0.5) * invTextureSize).rgb;
		height += v.x * scale * smoothing;
		slope += v.yz * scale * smoothing;

		scale *= 0.5;
		texCoordScale *= 2.0;
	}
}
void main() {
  vec2 iPosition = mix(ivec2(vPosition), ivec2(resolution-1) - ivec2(vPosition), flipAxis);

  vec2 tPosition = textureOffset + iPosition * textureStep;
  rawTexCoord = textureOffset + iPosition * textureStep;
  texCoord = (vec2(tPosition) + vec2(0.5)) / textureSize(heights, 0);

  float y = texture(heights, texCoord).r;
  vec2 slope = texture(slopes, texCoord).xy;
  compute_height_and_slope(y, slope);

  vec2 p = iPosition / vec2(resolution - 1);
  vec3 pos = vec3(p.x, y, p.y) * scale + position;
  gl_Position = modelViewProjection * vec4(pos, 1.0);
}
