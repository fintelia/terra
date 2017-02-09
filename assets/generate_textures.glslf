#version 450 core

in vec2 vTexCoord;
layout(pixel_center_integer) in vec4 gl_FragCoord;
out vec4 normals;

uniform sampler2D heights;
uniform float yScale;

void main() {
  ivec2 uv = ivec2(gl_FragCoord.xy);

  float hxm = texelFetchOffset(heights, uv, 0, ivec2(-1, 0)).r * yScale;
  float hxp = texelFetchOffset(heights, uv, 0, ivec2( 1, 0)).r * yScale;
  float hym = texelFetchOffset(heights, uv, 0, ivec2( 0,-1)).r * yScale;
  float hyp = texelFetchOffset(heights, uv, 0, ivec2( 0, 1)).r * yScale;

  vec3 normal = vec3(hxp - hxm, 2, hyp - hym);
  normals = vec4(normalize(normal),1);
}
