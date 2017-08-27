#line 2
in vec2 texCoord;
in vec3 fPosition;
out vec4 OutColor;

uniform sampler2D shadows;
uniform sampler2DArray materials;
uniform usampler2D splatmap;
uniform sampler2D colormap;

uniform vec3 eyePosition;

vec3 compute_color(vec3 position, vec2 slope) {
  vec3 normal = normalize(vec3(slope.x, 1.0, slope.y));

  float shadow_height = texture(shadows, texCoord).r;
  float shadow = smoothstep(shadow_height - 0.5, shadow_height + 20.5, position.y);

  float grass_amount = step(0.25, length(slope));
  float lod = textureQueryLOD(materials, position.xz * 0.1).x;
  vec3 rock_color = vec3(0.25,0.2,0.15);//vec3(.25, .1, .05) + vec3(0.1);
  vec3 grass_color = vec3(0.5);//textureLod(materials, vec3(position.xz * 0.1, 0), lod * 1.5).rgb;
  vec3 color = mix(grass_color, rock_color, grass_amount);
  float nDotL = max(dot(normalize(normal), normalize(vec3(0,1,1))), 0.0) * 0.8 + 0.2;
  //nDotL *= shadow * 0.98 + 0.02;

  float b = 0.01;
  float distance = distance(position, eyePosition);
  vec3 rayDir = normalize(position - eyePosition);
  float fogAmount = clamp(0.0005 * exp(-b*eyePosition.y) * (1.0 - exp(-b*rayDir.y*distance)) / (b*rayDir.y), 0, 1);
  vec3 fogColor = vec3(0.6, 0.6, 0.6);

  return mix(vec3(nDotL) * color, fogColor, fogAmount);
}


vec3 material(vec3 pos, uint mat) {
	if(mat == 0) return vec3(0.25,0.2,0.15);
	return vec3(0.5);
}
vec3 compute_splatting(vec3 pos, vec2 t) {
	t += 0.00001 * vec2(fractal(pos.xz), fractal(pos.xz + vec2(25)));

	vec2 weights = fract(t.xy * textureSize(splatmap, 0).xy - 0.5);
	uvec4 m = textureGather(splatmap, t, 0);
	vec4 w = mix(mix(vec4(0,0,0,1), vec4(1,0,0,0), weights.y),
				 mix(vec4(0,0,1,0), vec4(0,1,0,0), weights.y), weights.x);

	w = max(w - 0.15, 0);
	w /= w.x + w.y + w.z + w.w;

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
  compute_height_and_slope(fPosition.xz, texCoord, height, slope);
  vec3 position = vec3(fPosition.x, height, fPosition.z);

  vec2 t = position.xz / (2.0 * textureSize(splatmap, 0)) + vec2(0.5);
  if(t.x > 0 && t.x < 1 && t.y > 0 && t.y < 1) {
	  vec4 d = vec4(dFdx(position.xz * 0.5),
					dFdy(position.xz * 0.5));
	  float s = max(length(d.xz), length(d.yw));
	  if(s >= 1.0) {
		  OutColor.rgb = texture(colormap, t).rgb;
	  } else {
		  vec3 normal = normalize(vec3(slope.x, 1.0, slope.y));
		  float nDotL = max(dot(normalize(normal), normalize(vec3(0,1,1))), 0.0) * 0.8 + 0.2;
		  OutColor.rgb = compute_splatting(position, t) * nDotL;
	  }
  } else {
	  OutColor.rgb = vec3(0.5);
  }
}
