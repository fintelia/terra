#line 2
uniform sampler2DArray heights;
uniform int layer;
uniform vec2 layerCenter;
uniform vec2 parentCenter;
uniform float layerStep;

out float OutHeight;

void main() {
	vec2 center = textureSize(heights, 0).xy * 0.5;
	vec2 texCoord = gl_FragCoord.xy - vec2(0.5);

	vec2 parentTexCoord = (texCoord - center) * 0.5 + center;

	OutHeight = texture(heights, vec3((parentTexCoord + vec2(0.5)) / textureSize(heights, 0).xy, layer-1)).r;
}
