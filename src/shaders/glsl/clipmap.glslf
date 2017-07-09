#version 450 core

in vec2 rawTexCoord;
in vec2 texCoord;
out vec4 OutColor;

uniform sampler2D heights;
uniform sampler2D normals;
uniform sampler2D shadows;

uniform sampler2D detail;

/// Uses a fractal to refine the height and slope sourced from the course texture.
void compute_height_and_slope(inout float height, inout vec2 slope) {
	vec2 invTextureSize = 1.0  / textureSize(detail,0);

	float smoothing = mix(0.1, 1.0, smoothstep(0.0, 1.0, length(slope)));

	float scale = 10.0;
	float texCoordScale = 8.0;
	for(int i = 0; i < 6; i++) {
		vec3 v = texture(detail, (rawTexCoord * texCoordScale + 0.5) * invTextureSize).rgb;
		height += v.x * scale * smoothing;
		slope += v.yz * scale * smoothing;

		scale *= 0.5;
		texCoordScale *= 2.0;
	}
}

void main() {
  if(texCoord.x < 0 || texCoord.y < 0 || texCoord.x > 1 || texCoord.y > 1)
	  discard;

  float height = texture(heights, texCoord).x;
  vec3 normal = normalize(texture(normals, texCoord).rgb * vec3(2.0, 1.0, 2.0) - vec3(1.0, 0.0, 1.0));
  vec2 slope = normal.xz / normal.y;
  compute_height_and_slope(height, slope);
  normal = normalize(vec3(slope.x, 1.0, slope.y));

  float shadow_height = texture(shadows, texCoord).x;
  float shadow = 1.0;//clamp(1.0 + 1000 * (height - shadow_height), 0.2, 1.0);

  float grass_amount = smoothstep(0, 1, clamp(length(slope) * 2, 0, 1));

  vec3 rock_color = (vec3(.25, .1, .05) + vec3(.85, .72, .53)) * 0.5;
  vec3 grass_color = vec3(.85, .72, .53);//vec3(1);//vec3(0.0, 0.5, .1);
  vec3 color = mix(grass_color, rock_color, grass_amount);
  float nDotL = max(dot(normalize(normal), normalize(vec3(0,1,1))), 0.0) * 0.8 + 0.2;
  OutColor = vec4(shadow * vec3(nDotL) * color, 1);
}
