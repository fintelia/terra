#version 450 core
#include "declarations.glsl"

struct Vertex {
    vec3 position;
    float ao;
    vec3 lod_position;
    uint color;
    vec3 normal;
    float texcoord_u;
    vec3 binormal;
    float texcoord_v;
};
layout(std430, binding = 0) readonly buffer DataBlock {
    Vertex vertices[];
} model_storage;

layout(push_constant) uniform constants {
	float x;
    float z;
} push_constants;

layout(location = 0) out vec4 position;
layout(location = 1) out vec2 texcoord;
layout(location = 2) out vec3 normal;
layout(location = 3) out float ao;
layout(location = 4) out mat4 view;

void main() {
    Vertex vertex = model_storage.vertices[gl_VertexIndex];

    float y = 1.0 - max(abs(push_constants.x), abs(push_constants.z));

    vec3 up = vec3(0,1,0);
    vec3 f = normalize(vec3(push_constants.x, y, push_constants.z));

    if (y > 0.999)
        up = vec3(0, 0, 1);

    vec3 s = normalize(cross(f, up));
    vec3 u = cross(s, f);

    vec3 eye = vec3(0, 10, 0) - f * 5;

    mat4 proj = mat4(0.1,0,0,0, 0,0.1,0,0, 0,0,-0.1,0, 0,0,0,1);
    view = mat4(
        s.x, u.x, -f.x, 0,
        s.y, u.y, -f.y, 0,
        s.z, u.z, -f.z, 0,
        -dot(eye, s), -dot(eye, u), dot(eye, f), 1
    );

    position = proj * view * vec4(vertex.position.xzy, 1);

    texcoord = vec2(vertex.texcoord_u, vertex.texcoord_v);
    normal = vertex.normal;
    ao = vertex.ao;

    // position.z = 0.5;
    // position.y -= 1.0;

    gl_Position = position;
}