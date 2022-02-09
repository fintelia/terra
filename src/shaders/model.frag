#version 450 core
#include "declarations.glsl"

layout(early_fragment_tests) in;

layout(location = 0) in vec4 position;
layout(location = 1) in vec2 texcoord;
layout(location = 2) in vec3 normal;
layout(location = 3) in mat4 view;

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec2 out_normals;
layout(location = 2) out float out_depth;

void main() {
    out_color = texcoord.y < 0.5 ? vec4(0.2,0.1,0.1,1) : vec4(0.33*.13,0.57*.13,0.0,1);
    out_normals = (view * vec4(normal,0)).xz;
    out_depth = position.z * 5;
}
