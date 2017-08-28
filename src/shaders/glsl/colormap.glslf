#line 2
uniform int resolution;
uniform float textureSpacing;
uniform float heightsSpacing;
uniform int layer;

uniform sampler2DArray materials;

out int OutSplat;
out vec4 OutColor;

vec3 material(vec3 pos, uint mat) {
	return textureLod(materials, vec3(0.5, 0.5, mat), 1000).rgb * (1.0 + fractal2(pos.xz) * 0.2);
}

vec3 compute_color(vec3 position, vec2 slope) {
  vec3 normal = normalize(vec3(slope.x, 1.0, slope.y));

  if(length(slope) <= 0.25)
	  return material(position, 1);
  else
	  return material(position, 0);
  // float nDotL = max(dot(normalize(normal), normalize(vec3(0,1,1))), 0.0);
  // return color;// * nDotL;
}

void main() {
	vec2 pos = (gl_FragCoord.xy - vec2(resolution * 0.5)) * textureSpacing * exp2(layer);
	vec2 texCoord = pos / (heightsSpacing * textureSize(heights, 0)) + vec2(0.5);

	vec2 slope;
	float height;
	compute_height_and_slope(pos, texCoord, height, slope);

	OutColor = vec4(compute_color(vec3(pos.x, height, pos.y), slope), 0);
}
