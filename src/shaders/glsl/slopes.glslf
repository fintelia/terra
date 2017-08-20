#line 2
uniform sampler2DArray heights;
uniform int layer;
uniform float layerStep;

out vec2 OutSlope;

void main() {
	vec2 center = textureSize(heights, 0).xy * 0.5;
	ivec2 texCoord = ivec2(gl_FragCoord.xy - vec2(0.5));

	float mx = texelFetch(heights, ivec3(texCoord - ivec2(1,0), layer), 0).x;
	float px = texelFetch(heights, ivec3(texCoord + ivec2(1,0), layer), 0).x;
	float my = texelFetch(heights, ivec3(texCoord - ivec2(0,1), layer), 0).x;
	float py = texelFetch(heights, ivec3(texCoord + ivec2(0,1), layer), 0).x;

	OutSlope = vec2(px - mx, py - my) / (2.0 * layerStep);
}
