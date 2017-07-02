#version 450 core

in vec2 vTexCoord;
layout(pixel_center_integer) in vec4 gl_FragCoord;

out vec4 normals;
out float shadows;

uniform sampler2D heights;
uniform float yScale;

void main() {
  ivec2 uv = ivec2(gl_FragCoord.xy);

  float hraw = texelFetch(heights, uv, 0).r;

  float hxm = texelFetchOffset(heights, uv, 0, ivec2(-1, 0)).r * yScale;
  float hxp = texelFetchOffset(heights, uv, 0, ivec2( 1, 0)).r * yScale;
  float hym = texelFetchOffset(heights, uv, 0, ivec2( 0,-1)).r * yScale;
  float hyp = texelFetchOffset(heights, uv, 0, ivec2( 0, 1)).r * yScale;

  vec3 normal = vec3(hxp - hxm, 2.0, hyp - hym);
  normals = vec4(normalize(normal),1);
  normals.xz = normals.xz * 0.5 + vec2(0.5);

  float slope = 0.0003;
  shadows = 0;
  // for(int x = uv.x; x < textureSize(heights, 0).x; x++) {
  // 	  float h = texelFetch(heights, ivec2(x,uv.y), 0).r;
  // 	  shadows = max(shadows, h - slope * (x - uv.x));
  // }
}
