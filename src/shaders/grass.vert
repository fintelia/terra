#line 2

layout(set = 0, binding = 0) uniform UniformBlock {
    mat4 view_proj;
	vec3 camera;
	float padding;
} ubo;

layout(set = 0, binding = 1, std140) uniform NodeBlock {
	vec3 relative_position;
	float min_distance;
	vec3 parent_relative_position;
	float padding1;

    uint slot;
    uvec3 padding2;
} node;

struct Entry {
    vec3 position;
    float weight;
};
layout(std430, binding = 2) buffer DataBlock {
    Entry entries[][64*64];
} grass_storage;

layout(set = 0, binding = 2) uniform sampler linear;
layout(set = 0, binding = 9) uniform texture2DArray displacements;

layout(location = 0) out vec3 position;
layout(location = 1) out vec3 color;

void main() {
    Entry entry = grass_storage.entries[node.slot][gl_VertexIndex / 6];
    position = entry.position - node.relative_position;

    vec3 up = normalize(position + ubo.camera);

	float morph = 1 - smoothstep(0.9, 1, length(position) / node.min_distance);

    if (gl_VertexIndex % 6 == 0) position += morph * vec3(.02,0,0);
    if (gl_VertexIndex % 6 == 1) position += morph * vec3(-.02,0,0);
    if (gl_VertexIndex % 6 == 2) position += morph * up*.15;
    if (gl_VertexIndex % 6 == 3) position += morph * vec3(0,.02,0);
    if (gl_VertexIndex % 6 == 4) position += morph * vec3(0,-.02,0);
    if (gl_VertexIndex % 6 == 5) position += morph * up*.15;

    if (gl_VertexIndex % 3 == 0) color = vec3(0.05, 0.2, 0);
    if (gl_VertexIndex % 3 == 1) color = vec3(0.05, 0.2, 0);
    if (gl_VertexIndex % 3 == 2) color = vec3(0.05, 0.3, 0);

    gl_Position = ubo.view_proj * vec4(position, 1.0);
}