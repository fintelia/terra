#line 2
in vec2 texCoord;
in vec3 fPosition;
out vec4 OutColor;

uniform sampler2D shadows;
uniform sampler2DArray materials;

uniform vec3 eyePosition;

vec3 compute_color(float height, vec2 slope) {
  vec3 normal = normalize(vec3(slope.x, 1.0, slope.y));

  // float shadow_height = texture(shadows, texCoord).r;
  // float shadow = smoothstep(shadow_height - 0.5, shadow_height + 20.5, position.y);

  float grass_amount = step(0.25, length(slope));
  // float lod = textureQueryLOD(materials, position.xz * 0.1).x;
  vec3 rock_color = vec3(0.25,0.2,0.15);//vec3(.25, .1, .05) + vec3(0.1);
  vec3 grass_color = vec3(0.5);//textureLod(materials, vec3(position.xz * 0.1, 0), lod * 1.5).rgb;
  vec3 color = mix(grass_color, rock_color, grass_amount);
  float nDotL = max(dot(normal, normalize(vec3(0,1,1))), 0.0) * 0.8 + 0.2;
  //nDotL *= shadow * 0.98 + 0.02;

  return vec3(nDotL) * color;
}

void main() {
  if(texCoord.x < 0 || texCoord.y < 0 || texCoord.x > 1 || texCoord.y > 1)
	  discard;

  vec3 t = compute_texcoord(fPosition.xz);
  float height = texture(heights, t).r;
  vec2 slope = texture(slopes, t).xy;
  vec3 pos = vec3(fPosition.x, height, fPosition.z);

  vec2 weights = fract(t.xy * (textureSize(heights, 0).xy - 1) - 0.5);
  vec4 heightVals = vec4(0);//textureGather(heights, t, 0);
  vec4 slopeXVals = textureGather(slopes, t, 0);
  vec4 slopeYVals = textureGather(slopes, t, 1);

  vec3 color01 = compute_color(heightVals.x, vec2(slopeXVals.x, slopeYVals.x));
  vec3 color11 = compute_color(heightVals.y, vec2(slopeXVals.y, slopeYVals.y));
  vec3 color10 = compute_color(heightVals.z, vec2(slopeXVals.z, slopeYVals.z));
  vec3 color00 = compute_color(heightVals.w, vec2(slopeXVals.w, slopeYVals.w));
  vec3 color = mix(mix(color00, color01, weights.y),
				   mix(color10, color11, weights.y), weights.x);

  float dx = length(dFdx(t.xy * textureSize(heights, 0).xy));
  //  color = compute_color(height, slope);
  
  float b = 0.01;
  float distance = distance(pos, eyePosition);
  vec3 rayDir = normalize(pos - eyePosition);
  float fogAmount = 0;//clamp(0.0005 * exp(-b*eyePosition.y) * (1.0 - exp(-b*rayDir.y*distance)) / (b*rayDir.y), 0, 1);
  vec3 fogColor = vec3(0.6, 0.6, 0.6);

  OutColor = vec4(mix(color, fogColor, fogAmount), 1);

  if(dx < 1.0)
	  OutColor.r *= 3;
}
