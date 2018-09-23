#line 2

uniform mat4 modelViewProjection;

uniform sampler2DArray colors;
uniform sampler2DArray normals;
uniform sampler2DArray splats;
uniform sampler2DArray materials;
uniform samplerCube sky;

uniform sampler2DArray oceanSurface;
uniform sampler2D noise;
uniform float noiseWavelength;
uniform vec3 cameraPosition;
uniform vec3 sunDirection;

in vec3 fPosition;
in vec2 fTexcoord;
in vec2 fParentTexcoord;
in vec2 fColorsLayer;
in vec2 fNormalsLayer;
in vec2 fSplatsLayer;
in float fMorph;

out vec4 OutColor;

vec4 fractal(vec2 pos) {
	vec4 value = vec4(0.0);
	float scale = 0.5;
	float wavelength = 2.0;
	for(int i = 0; i < 5; i++) {
		vec4 v = texture(noise, pos * noiseWavelength / wavelength) * 6 - 3;
		value += v * scale;
		scale *= 0.5;
		wavelength *= 0.5;
	}
	return value;
}

float fractal2(vec2 pos) {
	// if(texture(noise, pos / 32).x > 0.5)
	// 	return 0;
	// return 1;

	//	if(fract(pos.y/32) < 0.01) return 100;
	return (/*texture(noise, pos / (32*32*32)).x
			  + */texture(noise, pos / (32*32)).x
			+ texture(noise, pos / 32).x) / 2;
}

vec3 aerial_perspective(vec3 color, vec3 position) {
	vec3 air = atmosphere(cameraPosition, position, sunDirection);
    return 1.0 - exp(-0.75 * air) + color;
}

vec3 material(vec3 pos, uint mat) {
	// TODO: remove hacks here
	vec3 v = texture(materials, vec3(pos.xz/8 , mat), 0).rgb;
	if(mat == 3) v *= 0.5;
	if(mat == 4) v *= 0.4;
	// if(mat == 0) v *= vec3(.5,.8,.8);
	return v * (0.625 + fractal2(pos.xz) * .75);
}

vec3 compute_splatting(vec3 pos, vec2 texcoord, vec3 normal, float splatsLayer) {
	texcoord += 0.0001 * fractal(pos.xz).xy * 10;
	vec2 weights = fract(texcoord.xy * textureSize(splats, 0).xy - 0.5);
	uvec4 m = uvec4(ceil(textureGather(splats, vec3(texcoord, splatsLayer), 3) * 255));
	vec4 w = mix(mix(vec4(0,0,0,1), vec4(1,0,0,0), weights.y),
				 mix(vec4(0,0,1,0), vec4(0,1,0,0), weights.y), weights.x);

	return material(pos, m.x) * w.x +
		material(pos, m.y) * w.y +
		material(pos, m.z) * w.z +
		material(pos, m.w) * w.w;
}

vec3 water_color() {
	vec3 n = normalize(fPosition + vec3(0,planetRadius,0)); // y
	vec3 t = normalize(cross(vec3(0,0,1), n)); // x
	vec3 b = normalize(cross(t, n)); // z

	vec3 ray = normalize(fPosition - cameraPosition);
	vec3 normal = mat3(t, n, b) * normalize(texture(oceanSurface, vec3(fPosition.xz * 0.000001,0)).xzy * 2 - 1);
	vec3 reflected = reflect(ray, normalize(normal));

	vec3 reflectedColor = vec3(0,0.1,0.2) * 0.27;//textureLod(sky, normalize(reflected), 4).rgb;
	vec3 refractedColor = vec3(0,0.1,0.2) * 0.25;

	float R0 = pow(0.333 / 2.333, 2);
	float R = R0 + (1 - R0) * pow(1 - reflected.y, 5);
	return mix(refractedColor, reflectedColor, R);
}

vec3 land_color(vec2 texcoord, float colorsLayer, float normalsLayer, float splatsLayer) {
	vec4 color = texture(colors, vec3(texcoord, colorsLayer));
	vec3 normal = normalize(texture(normals, vec3(texcoord, normalsLayer)).xyz * 2.0 - 1.0);
	normal = normalize(fPosition + vec3(0,planetRadius,0));

	if(splatsLayer >= 0) {
		color.rgb = compute_splatting(fPosition, texcoord, normal, splatsLayer);
	}

	color.rgb = mix(color.rgb, water_color(), smoothstep(0.2,0.5, color.a));
	color.rgb *= max(dot(normal, sunDirection), 0);
	return color.rgb;
}

void main() {
	OutColor = vec4(0,0,0,1);
	if(fMorph < 0.9999) {
		OutColor.rgb += land_color(fParentTexcoord, fColorsLayer.y, fNormalsLayer.y, fSplatsLayer.y)
			* (1.0 - fMorph);
	}
	if(fMorph > 0.0) {
		OutColor.rgb += land_color(fTexcoord, fColorsLayer.x, fNormalsLayer.x, fSplatsLayer.x)
			* fMorph;
	}

	OutColor.rgb = precomputed_aerial_perspective(OutColor.rgb, fPosition, cameraPosition, sunDirection);
	// if(length(fPosition.xz) > 3000) {
	// 	OutColor.rgb *= vec3(2,.5,.5);
	// 	if(length(fPosition.xz) < 3030)
	// 		OutColor.rgb += vec3(.0,-.1,-.1);
	// } else if(fract(fPosition.x) < 0.05 || fract(fPosition.z) < 0.05) {
	// 	OutColor.rgb = vec3(.15);
	// }
 	OutColor.rgb += dither();
}
