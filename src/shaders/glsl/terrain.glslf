#line 2

uniform mat4 modelViewProjection;

uniform sampler2DArray colors;
uniform sampler2DArray normals;
uniform sampler2DArray water;
uniform sampler2DArray materials;
uniform samplerCube sky;

uniform sampler2DArray oceanSurface;
uniform sampler2D noise;
uniform float noiseWavelength;
uniform vec3 cameraPosition;
uniform vec3 sunDirection;

in vec3 fPosition;
in vec2 fTexcoord;
in float fColorsLayer;
in float fNormalsLayer;
in float fWaterLayer;

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
	float value = 0.0;
	float scale = 1.0 / 10;
	float wavelength = 64.0;
	for(int i = 0; i < 10; i++) {
		vec3 v = texture(noise, pos * noiseWavelength / wavelength + vec2(0.123 * i)).rgb;
		value += v.x * scale;
		// scale *= 0.5;
		wavelength *= 0.5;
	}
	return value;
}

vec3 aerial_perspective(vec3 color, vec3 position) {
	vec3 air = atmosphere(cameraPosition, position, sunDirection);
    return 1.0 - exp(-0.75 * air) + color;
}

vec3 material(vec3 pos, uint mat) {
	return texture(materials, vec3(pos.xz * 0.5, mat)).rgb;// * (1.0 + fractal2(pos.xz) * 0.2);
}

vec3 compute_splatting(vec3 pos, vec2 t) {
	//	t += 0.0001 * fractal(pos.xz).xy * 10;

	vec2 weights = fract(t.xy * textureSize(normals, 0).xy - 0.5);
	uvec4 m = uvec4(ceil(textureGather(normals, vec3(t, fNormalsLayer), 3) * 255));
	vec4 w = mix(mix(vec4(0,0,0,1), vec4(1,0,0,0), weights.y),
				 mix(vec4(0,0,1,0), vec4(0,1,0,0), weights.y), weights.x);

	return material(pos, m.x) * w.x +
		material(pos, m.y) * w.y +
		material(pos, m.z) * w.z +
		material(pos, m.w) * w.w;
}
void main() {
	if(fNormalsLayer >= 0) {
		vec3 normal = normalize(texture(normals, vec3(fTexcoord, fNormalsLayer)).xyz * 2.0 - 1.0);

		OutColor.rgb = compute_splatting(fPosition, fTexcoord);
		OutColor.rgb *= dot(normal, normalize(vec3(0,1,1)));
	} else {
		vec4 color = texture(colors, vec3(fTexcoord, fColorsLayer));

		OutColor = vec4(color.rgb, 1);
		OutColor.rgb *= color.a;
	}

	float waterAmount = texture(water, vec3(fTexcoord, fWaterLayer)).x;
	if(waterAmount > 0) {
		vec3 ray = normalize(fPosition - cameraPosition);
		vec3 normal = texture(oceanSurface, vec3(fPosition.xz * 0.001, 0)).xzy * 2 - 1;
		vec3 reflected = reflect(ray, normalize(normal));

		vec3 reflectedColor = vec3(0,0.05,0.1);//textureLod(sky, normalize(reflected), 5).rgb;
		vec3 refractedColor = vec3(0,0.1,0.2)*0.5;

		float R0 = pow(0.333 / 2.333, 2);
		float R = R0 + (1 - R0) * pow(1 - reflected.y, 5);
		vec3 waterColor = mix(refractedColor, reflectedColor, R);
		OutColor.rgb = mix(OutColor.rgb, waterColor, waterAmount);
	}

	OutColor.rgb = aerial_perspective(OutColor.rgb, fPosition);
	// if(fract(fPosition.x * 0.001) < 0.01 || fract(fPosition.z * 0.001) < 0.01)
	// 	OutColor.rgb = vec3(0);
}
