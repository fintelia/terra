#version 450 core

in vec2 texCoord;
out vec4 OutColor;

uniform sampler2D heights;
uniform sampler2D normals;
uniform sampler2D shadows;

uniform sampler2D detail_normals;

/// Uses a fractal to refine the height and slope sourced from the course texture.
vec4 get_normal_and_height(vec2 interpolated_slope, float interpolated_height) {
	vec2 slope = interpolated_slope;
	float height = interpolated_height;

	float scale = 1.0;
	float texCoordScale = 1.0;
	for(int i = 0; i < 6; i++) {
		float smoothing = smoothstep(0, 1, clamp(length(slope), 0, 1));

		vec3 v = texture(detail_normals, texCoord * texCoordScale).rgb;
		height += v.x * scale * smoothing;
		slope += v.yz * scale * smoothing;

		scale *= 0.5;
		texCoordScale *= 2.0;
	}

	return vec4(normalize(vec3(slope.x, 1.0, slope.y)), height);
}

void main() {
  float height = texture(heights, texCoord).x;
  float shadow_height = texture(shadows, texCoord).x;

  float shadow = 1.0;//clamp(1.0 + 1000 * (height - shadow_height), 0.2, 1.0);

  vec3 normal = normalize(texture(normals, texCoord).rgb);
  vec2 slope = normal.xz / normal.y;
  vec4 normal_and_height = get_normal_and_height(slope, height);
  normal = normal_and_height.xyz;

  vec3 color = vec3(.8, .65, .5);
  float nDotL = dot(normalize(normal), normalize(vec3(0,1,1)));
  OutColor = vec4(shadow * vec3(nDotL) * color, 1);
}
