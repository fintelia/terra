#version 450 core
#include "declarations.glsl"

layout(set = 0, binding = 0, std140) uniform UniformBlock {
    Globals globals;
};

layout(set = 0, binding = 8, std430) readonly buffer NodeSlots {
	Node nodes[];
};

struct Entry {
    vec3 position;
    float angle;
    vec3 albedo;
    float slant;
    vec2 texcoord;
    vec2 _padding1;
    vec4 _padding2;
};
layout(std430, binding = 2) readonly buffer DataBlock {
    Entry entries[][32*32];
} grass_storage;

layout(set = 0, binding = 3) uniform sampler linear;
// layout(set = 0, binding = 9) uniform texture2DArray displacements;

layout(location = 0) out vec3 position;
layout(location = 1) out vec3 color;
layout(location = 2) out vec2 texcoord;
layout(location = 3) out vec3 normal;

const vec3 tangents[6] = vec3[6](
	vec3(0,1,0),
	vec3(0,-1,0),
	vec3(1,0,0),
	vec3(-1,0,0),
	vec3(1,0,0),
	vec3(-1,0,0)
);

void main() {
    uint entry_index = gl_VertexIndex / 7;
    uint index = gl_VertexIndex % 7;
    uint slot = gl_InstanceIndex / 16;

    Node node = nodes[slot];
    Entry entry = grass_storage.entries[(slot - GRASS_BASE_SLOT) * 16 + gl_InstanceIndex % 16][entry_index];
    position = entry.position - node.relative_position;

    vec3 up = normalize(position + globals.camera);
	vec3 bitangent = normalize(cross(up, tangents[node.face]));
	vec3 tangent = normalize(cross(up, bitangent));

	float morph = 1 - smoothstep(0.7, .99, length(position) / node.min_distance);

    vec3 offset;
    float width = 0.01;
    float height = 0.1;

    if (node.min_distance > 24) {
        width *= mix(1, 1.5, smoothstep(0.7, .99, 4 * length(position) / node.min_distance));
        width *= mix(1, 1.5, smoothstep(0.7, .99, 2 * length(position) / node.min_distance));

        height *= mix(1, 1.5, smoothstep(0.7, .99, 4 * length(position) / node.min_distance));
        height *= mix(1, 1.5, smoothstep(0.7, .99, 2 * length(position) / node.min_distance));
        //morph *= smoothstep(0.7, .99, 2 * length(position) / node.min_distance);
    } else if (node.min_distance > 12) {
        width *= mix(1, 1.5, smoothstep(0.7, .99, 2 * length(position) / node.min_distance));
        height *= mix(1, 1.5, smoothstep(0.7, .99, 2 * length(position) / node.min_distance));
    }

    vec2 uv = vec2(0);
    if (index == 0) uv = vec2(-1, 0);
    else if (index == 1) uv = vec2(1, 0);
    else if (index == 2) uv = vec2(-.9, .3);
    else if (index == 3) uv = vec2(.9, .3);
    else if (index == 4) uv = vec2(-.7, .6); 
    else if (index == 5) uv = vec2(.7, .6); 
    else if (index == 6) uv = vec2(0, 1); 

    vec3 u = cos(entry.angle) * tangent + sin(entry.angle) * bitangent;
    vec3 w = -sin(entry.angle) * tangent + cos(entry.angle) * bitangent;
    position += (u*width*uv.x + (up + w*uv.y*entry.slant)*height*uv.y) * morph;

    color = mix(entry.albedo, vec3(0, .4, .01), .0*uv.y);
    texcoord = entry.texcoord;
    normal = normalize(w + up);

    gl_Position = globals.view_proj * vec4(position, 1.0);
}