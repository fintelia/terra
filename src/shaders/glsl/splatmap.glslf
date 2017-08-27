#line 2
uniform int resolution;
uniform float textureSpacing;
uniform float heightsSpacing;

out int OutSplat;
out vec4 OutColor;


vec3 compute_color(vec3 position, vec2 slope) {
  vec3 normal = normalize(vec3(slope.x, 1.0, slope.y));

  float grass_amount = step(0.25, length(slope));
  vec3 rock_color = vec3(0.25,0.2,0.15);
  vec3 grass_color = vec3(0.5);
  vec3 color = mix(grass_color, rock_color, grass_amount);
  float nDotL = max(dot(normalize(normal), normalize(vec3(0,1,1))), 0.0) * 0.8 + 0.2;

  return color;// * nDotL;
}

void main() {
	vec2 pos = (gl_FragCoord.xy - vec2(resolution * 0.5)) * textureSpacing;
	vec2 texCoord = pos / (heightsSpacing * textureSize(heights, 0)) + vec2(0.5);

	vec2 slope;
	float height;
	compute_height_and_slope(pos, texCoord, height, slope);

	OutSplat = 0;
	if(length(slope) < 0.25)
		OutSplat = 1;
	OutColor = vec4(compute_color(vec3(pos.x, height, pos.y), slope), 0);
}
