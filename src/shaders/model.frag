#version 450 core
#include "declarations.glsl"

layout(early_fragment_tests) in;

layout(location = 0) in vec4 position;
layout(location = 1) in vec2 texcoord;
layout(location = 2) in vec3 normal;
layout(location = 3) in float ao;
layout(location = 4) in mat4 view;

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec2 out_normals;
layout(location = 2) out float out_depth;

layout(set = 0, binding = 1) uniform sampler linear_wrap;
layout(set = 0, binding = 2) uniform texture2D models_albedo;


void main() {
    out_color = texture(sampler2D(models_albedo, linear_wrap), texcoord);
    if (out_color.a < .5)
        discard;
    out_color.rgb *= ao;
    out_color.a = 1;
    //out_color.rgb = mix(vec3(1,0,0), out_color.rgb, out_color.a);
    
    out_normals = (view * vec4(normal,0)).xz;
    out_depth = position.z * 5;
}
