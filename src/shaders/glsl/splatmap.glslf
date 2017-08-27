#line 2
uniform int resolution;
uniform float textureSpacing;
uniform float heightsSpacing;

out int OutSplat;
out vec4 OutColor;

void main() {
	vec2 pos = (gl_FragCoord.xy - vec2(resolution * 0.5)) * textureSpacing;
	vec2 texCoord = pos / (heightsSpacing * textureSize(heights, 0)) + vec2(0.5);

	vec2 slope;
	float height;
	compute_height_and_slope(pos, texCoord, height, slope);

	OutSplat = 0;
	if(length(slope) < 0.25)
		OutSplat = 1;
}
