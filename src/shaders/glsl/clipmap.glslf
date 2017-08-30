#line 2
in vec2 texCoord;
in vec3 fPosition;
out vec4 OutColor;

uniform sampler2D shadows;
uniform sampler2DArray materials;
uniform usampler2D splatmap;
uniform sampler2DArray colormap;

uniform vec3 eyePosition;

float compute_fog(vec3 position) {
	float b = 0.01;
	float distance = distance(position, eyePosition);
	vec3 rayDir = normalize(position - eyePosition);
	return clamp(0.0005 * exp(-b*eyePosition.y) * (1.0 - exp(-b*rayDir.y*distance)) / (b*rayDir.y), 0, 0.3);
}

vec3 material(vec3 pos, uint mat) {
	return texture(materials, vec3(pos.xz * 0.5, mat)).rgb * (1.0 + fractal2(pos.xz) * 0.2);
}

// void compare_weights(uvec2 mats, vec2 weights) {
// 	if(mats.x == mats.y) {
// 		weights.x += weights.y;
// 		weights.y = 0;
// 	}
// }

vec3 compute_splatting(vec3 pos, vec2 t) {
	t += 0.0001 * vec2(fractal(pos.xz), fractal(pos.xz + vec2(25)));

	vec2 weights = fract(t.xy * textureSize(splatmap, 0).xy - 0.5);
	uvec4 m = textureGather(splatmap, t, 0);
	vec4 w = mix(mix(vec4(0,0,0,1), vec4(1,0,0,0), weights.y),
				 mix(vec4(0,0,1,0), vec4(0,1,0,0), weights.y), weights.x);

	// compare_weights(m.xy, w.xy);
	// compare_weights(m.xz, w.xz);
	// compare_weights(m.xw, w.xw);

	// compare_weights(m.yz, w.yz);
	// compare_weights(m.yw, w.yw);
	// compare_weights(m.zw, w.zw);

	return material(pos, m.x) * w.x +
		material(pos, m.y) * w.y +
		material(pos, m.z) * w.z +
		material(pos, m.w) * w.w;
}

void main() {
	if(texCoord.x < 0 || texCoord.y < 0 || texCoord.x > 1 || texCoord.y > 1)
		discard;

	vec2 slope;
	float height;
	vec2 coarse_slope = texture(slopes, texCoord).xy;
	compute_height_and_slope(fPosition.xz, texCoord, height, slope);
	vec3 position = vec3(fPosition.x, height, fPosition.z);

	vec2 t = position.xz / (2.0 * textureSize(splatmap, 0)) + vec2(0.5);
	vec4 d = vec4(dFdx(position.xz * 0.5),
				  dFdy(position.xz * 0.5));
	float level = 0.5 * log2(min(dot(d.xz, d.xz), dot(d.yw, d.yw)));
	float maxLevel = textureSize(colormap, 0).z - 1.0;

	level = floor(level);
	float centerDistance = max(abs(t.x-0.5), abs(t.y-0.5));
	while(level < maxLevel && centerDistance > 0.5 * exp2(max(ceil(level),0))) {
		level = floor(level) + 1.0;
	}

	if(level >= maxLevel) {
		t = (t - 0.5) * exp2(-maxLevel) + vec2(0.5);
		OutColor.rgb = texture(colormap, vec3(t, maxLevel)).rgb;
	} else if(level >= 0.0) {
		float fLevel = floor(level);
		vec2 t1 = (t - 0.5) * exp2(-fLevel) + vec2(0.5);
		vec2 t2 = (t - 0.5) * exp2(-fLevel - 1) + vec2(0.5);
		vec3 color1 = textureLod(colormap, vec3(t1, fLevel),0).rgb;
		vec3 color2 = textureLod(colormap, vec3(t2, fLevel+1),0).rgb;
		OutColor.rgb = mix(color1, color2, level - fLevel);
	} else {
		vec3 coarse_normal = normalize(vec3(coarse_slope.x, 1.0, coarse_slope.y));
		vec3 normal = normalize(vec3(slope.x, 1.0, slope.y));
		float nDotL = max(dot(coarse_normal, normalize(vec3(0,1,1))), 0.0);
		OutColor.rgb = compute_splatting(position, t) * nDotL;

		if(level >= -1.0) {
			OutColor.rgb = mix(OutColor.rgb, textureLod(colormap, vec3(t,0), 0).rgb, level + 1);
		}
	}

	OutColor.rgb = mix(OutColor.rgb, vec3(0.6), compute_fog(position));
}
