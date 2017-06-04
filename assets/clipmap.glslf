#version 450 core

in vec2 texCoord;
out vec4 OutColor;

uniform sampler2D heights;
uniform sampler2D normals;
uniform sampler2D shadows;

void main() {
  float height = texture(heights, texCoord).x;
  float shadow_height = texture(shadows, texCoord).x;

  float shadow = clamp(1.0 + 1000 * (height - shadow_height), 0.2, 1.0);

  vec3 color = vec3(.8, .65, .5);
  vec3 normal = texture(normals, texCoord).rgb;
  float nDotL = dot(normalize(normal), normalize(vec3(0,1,1)));
  OutColor = vec4(shadow * vec3(nDotL) * color, 1);
  OutColor.rgb = vec3(0);
}
