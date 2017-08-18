in vec2 rawTexCoord;
in vec2 texCoord;
in vec3 fPosition;
out vec4 OutColor;

uniform sampler2D heights;
uniform sampler2D slopes;
uniform sampler2D shadows;
uniform sampler2DArray materials;

uniform vec3 eyePosition;

void main() {
  if(texCoord.x < 0 || texCoord.y < 0 || texCoord.x > 1 || texCoord.y > 1)
	  discard;

  float height = texture(heights, texCoord).x;
  vec2 slope = texture(slopes, texCoord).xy;
  compute_height_and_slope(fPosition.xz, height, slope);
  vec3 normal = normalize(vec3(slope.x, 1.0, slope.y));
  vec3 position = vec3(fPosition.x, height, fPosition.z);

  float shadow_height = texture(shadows, texCoord).r;
  float shadow = smoothstep(shadow_height - 0.5, shadow_height + 20.5, height);

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
  color = mix(vec3(nDotL) * color, fogColor, fogAmount);

  OutColor = vec4(color, 1);
}
