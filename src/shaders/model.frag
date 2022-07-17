#version 450 core
#include "declarations.glsl"

layout(early_fragment_tests) in;

layout(location = 0) in vec4 position;
layout(location = 1) in vec2 texcoord;
layout(location = 2) in vec3 normal;
layout(location = 3) in float ao;
layout(location = 4) in vec4 view_0;
layout(location = 5) in vec4 view_1;
layout(location = 6) in vec4 view_2;

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec4 out_normals;
layout(location = 2) out float out_depth;
layout(location = 3) out float out_ao;

layout(set = 0, binding = 1) uniform sampler linear_wrap;
layout(set = 0, binding = 2) uniform texture2D models_albedo;

void main() {
    mat4 view;
    view[0] = view_0;
    view[1] = view_1;
    view[2] = view_2;
    view[3] = vec4(0.0, 0.0, 0.0, 1.0);

    out_color = texture(sampler2D(models_albedo, linear_wrap), texcoord);
    if (out_color.a < 1 || gl_FragCoord.z == 0)
        discard;
    out_color.a = 1;//out_color.a > 0.5 ? 1 : 0;

    out_normals = vec4(normalize((view * vec4(normal,0)).xyz).xyz, 0);
    out_depth = gl_FragCoord.z;//position.z * 5;
    out_ao = 1 - ao;
}
