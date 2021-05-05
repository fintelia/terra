#line 2

layout(set = 0, binding = 0, std140) uniform UniformBlock {
    Globals globals;
};

layout(set = 0, binding = 1, std140) uniform NodeBlock {
	vec3 relative_position;
	float min_distance;
	vec3 parent_relative_position;
	float padding1;

    uint slot;
    uint face;
    uvec2 padding2;
} node;

struct Entry {
    vec4 position_u;
    vec4 albedo_v;
};
layout(std430, binding = 2) buffer DataBlock {
    Entry entries[][128*128];
} grass_storage;

layout(set = 0, binding = 3) uniform sampler linear;
// layout(set = 0, binding = 9) uniform texture2DArray displacements;

layout(location = 0) out vec3 position;
layout(location = 1) out vec3 color;
layout(location = 2) out vec2 texcoord;

const vec3 tangents[6] = vec3[6](
	vec3(0,1,0),
	vec3(0,-1,0),
	vec3(1,0,0),
	vec3(-1,0,0),
	vec3(1,0,0),
	vec3(-1,0,0)
);

void main() {
    Entry entry = grass_storage.entries[node.slot][gl_VertexIndex / 6];
    position = entry.position_u.xyz - node.relative_position;

    vec3 up = normalize(position + globals.camera);
	vec3 bitangent = normalize(cross(up, tangents[node.face]));
	vec3 tangent = normalize(cross(up, bitangent));

	float morph = 1 - smoothstep(0.5, .99, length(position) / node.min_distance);

    vec3 offset;
    float width = 0.01;
    float height = 0.05;

    if (gl_VertexIndex % 6 == 0) offset = bitangent * width;
    if (gl_VertexIndex % 6 == 1) offset = -bitangent * width;
    if (gl_VertexIndex % 6 == 2) offset = up*height;
    if (gl_VertexIndex % 6 == 3) offset = tangent * width;
    if (gl_VertexIndex % 6 == 4) offset = -tangent * width;
    if (gl_VertexIndex % 6 == 5) offset = up*height;

    position += offset * morph;

    color = entry.albedo_v.rgb;
    texcoord = vec2(entry.position_u.w, entry.albedo_v.w);

    gl_Position = globals.view_proj * vec4(position, 1.0);
}