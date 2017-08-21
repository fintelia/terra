#line 2
in vec2 texCoord;
in vec3 fPosition;
out vec4 OutColor;

uniform sampler2D shadows;
uniform sampler2DArray materials;

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

void main() {
  if(texCoord.x < 0 || texCoord.y < 0 || texCoord.x > 1 || texCoord.y > 1)
	  discard;

  vec2 slope;
  float height;
  vec3 c = compute_height_and_slope(fPosition.xz, texCoord, height, slope);
  vec3 position = vec3(fPosition.x, height, fPosition.z);
  vec3 color = compute_color(position, slope);

  OutColor = vec4(color, 1);
//  OutColor.rgb = slope.rrg;
  OutColor.rgb = mix(OutColor.rgb, c * 0.5, 0.2);
}
