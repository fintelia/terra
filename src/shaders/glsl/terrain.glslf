#line 2
uniform mat4 modelViewProjection;

uniform sampler2DArray colors;
uniform sampler2DArray normals;
uniform sampler2DArray materials;

uniform sampler2D noise;
uniform float noiseWavelength;

in vec3 fPosition;
in vec2 fTexcoord;
in float fColorsLayer;
in float fNormalsLayer;

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

// float compute_fog(vec3 position) {
// 	float b = 0.01;
// 	float distance = distance(position, eyePosition);
// 	vec3 rayDir = normalize(position - eyePosition);
// 	return clamp(0.0005 * exp(-b*eyePosition.y) * (1.0 - exp(-b*rayDir.y*distance)) / (b*rayDir.y), 0, 0.3);
// }

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
}
