#version 450 core

in vec2 teTexCoord;
out vec4 OutColor;

uniform sampler2D heights;
uniform sampler2D normals;

void main() {
  vec3 color = vec3(.8, .65, .5);
  vec3 normal = texture(normals, teTexCoord).rgb;
  float nDotL = dot(normalize(normal), normalize(vec3(0,1,1)));
  OutColor = vec4(vec3(nDotL) * color, 1);
}
