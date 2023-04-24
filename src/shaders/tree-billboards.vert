#version 450 core
#include "declarations.glsl"

layout(set = 0, binding = 0, std140) uniform UniformBlock {
    Globals globals;
};

layout(set = 0, binding = 8, std140) readonly buffer Nodes {
	Node nodes[];
};
layout(set = 0, binding = 1) uniform texture2DArray aerial_perspective;

struct Entry {
    vec3 position;
    uint albedo;
    float angle;
    float height;
    uint uv;
    float padding0;
};
layout(std430, binding = 2) readonly buffer DataBlock {
    Entry entries[];
} tree_billboards_storage;

layout(set = 0, binding = 3) uniform sampler linear;
// layout(set = 0, binding = 9) uniform texture2DArray displacements;

layout(location = 0) out vec3 position;
layout(location = 1) out vec3 color;
layout(location = 2) out vec2 texcoord;
layout(location = 3) out vec3 normal;
layout(location = 4) flat sample out uint slot;
layout(location = 5) out vec3 right;
layout(location = 6) out vec3 up;
layout(location = 7) out vec4 atmosphere;

const vec3 tangents[6] = vec3[6](
	vec3(0,1,0),
	vec3(0,-1,0),
	vec3(1,0,0),
	vec3(-1,0,0),
	vec3(1,0,0),
	vec3(-1,0,0)
);

void main() {
    uint entry_index = gl_VertexIndex / 4;
    uint index = gl_VertexIndex % 4;
    slot = gl_InstanceIndex / 16;

    Node node = nodes[slot];
    Entry entry = tree_billboards_storage.entries[((slot - TREE_BILLBOARDS_BASE_SLOT) * 16 + gl_InstanceIndex % 16) * 16384 + entry_index];
    position = entry.position - node.relative_position;

    up = normalize(position + globals.camera);
	vec3 bitangent = normalize(cross(up, tangents[node.face]));
	vec3 tangent = normalize(cross(up, bitangent));

	float morph = 1 - smoothstep(0.7, .99, length(position) / node.min_distance);

    vec2 uv = vec2(0);
    if (index == 0) uv = vec2(0, 0);
    else if (index == 1) uv = vec2(1, 0);
    else if (index == 2) uv = vec2(0, 1);
    else if (index == 3) uv = vec2(1, 1);

    // if (index == 0) uv = vec2(-4, -.10);
    // else if (index == 1) uv = vec2(4, -.10);
    // else if (index == 2) uv = vec2(-1, 1);
    // else if (index == 3) uv = vec2(1, 1);

    right = normalize(cross(position, up));

    if (morph > 0)
        position += 30*(up * (1-uv.y) + right * (uv.x-0.5));

    color = unpackUnorm4x8(entry.albedo).rgb;//vec3(0.33,0.57,0.0)*.13;
    texcoord = uv;
    normal = normalize(cross(right, up));
    atmosphere = textureLod(sampler2DArray(aerial_perspective, linear),
        layer_texcoord(node.layers[AERIAL_PERSPECTIVE_LAYER], unpackUnorm2x16(entry.uv)), 0);

    gl_Position = globals.view_proj * vec4(position, 1.0);
}