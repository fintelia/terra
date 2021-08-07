#version 450 core
#include "declarations.glsl"

layout(set = 0, binding = 0, std140) uniform UniformBlock {
    Globals globals;
};
// layout(set = 0, binding = 1, std140) readonly buffer NodeBlock {
// 	NodeState nodes[];
// };
layout(set = 0, binding = 8) uniform texture2DArray displacements;

layout(set = 0, binding = 10, std430) readonly buffer NodeSlots {
	NodeSlot node_slots[];
};

layout(location = 0) out vec3 out_position;
layout(location = 1) out vec2 out_texcoord;
layout(location = 2) out float out_morph;
layout(location = 3) out vec3 out_normal;
layout(location = 4) out vec3 out_tangent;
layout(location = 5) out vec3 out_bitangent;
layout(location = 6) out vec2 out_i_position;
layout(location = 7) flat out uint out_instance;

const vec3 tangents[6] = vec3[6](
	vec3(0,1,0),
	vec3(0,-1,0),
	vec3(1,0,0),
	vec3(-1,0,0),
	vec3(1,0,0),
	vec3(-1,0,0)
);

vec3 sample_displacements(vec3 texcoord) {
	vec2 t = texcoord.xy * textureSize(displacements, 0).xy - 0.5;
	vec2 f = fract(t);
	vec4 w = vec4(f.x * (1-f.y), (1-f.x)*(1-f.y), (1-f.x)*f.y, f.x * f.y);
	return texelFetch(displacements, ivec3(t, texcoord.z), 0).xyz* (1-f.x) * (1-f.y)
		+ texelFetch(displacements, ivec3(t+ivec2(1,0), texcoord.z), 0).xyz * (f.x) * (1-f.y)
		+ texelFetch(displacements, ivec3(t+ivec2(1,1), texcoord.z), 0).xyz * (f.x) * (f.y)
		+ texelFetch(displacements, ivec3(t+ivec2(0,1), texcoord.z), 0).xyz * (1-f.x) * (f.y);
}

void main() {
	uint resolution = 64;//nodes[gl_InstanceIndex].resolution;
	uvec2 base_origin = uvec2(0);//nodes[gl_InstanceIndex].base_origin;
	NodeSlot node = node_slots[gl_InstanceIndex];

	ivec2 iPosition = ivec2((gl_VertexIndex) % (resolution+1),
							(gl_VertexIndex) / (resolution+1)) + ivec2(base_origin);

	int displacements_slot = node.layer_slots[DISPLACEMENTS_LAYER];
	vec3 texcoord = vec3(node.layer_origins[DISPLACEMENTS_LAYER] + vec2(iPosition) * node.layer_steps[DISPLACEMENTS_LAYER], displacements_slot); //vec3(0.5 / 65.0 + desc.origin * (64.0 / 65.0), desc.slot) + vec3(vec2(iPosition) / 64.0 * pow(0.5, node.layers[DISPLACEMENTS_LAYER]), 0);
	vec3 position = sample_displacements(texcoord) - node_slots[displacements_slot].relative_position;

	float morph = 1 - smoothstep(0.9, 1, length(position) / node.min_distance);
	vec2 nPosition = mix(vec2((iPosition / 2) * 2), vec2(iPosition), morph);

	if (morph < 1.0) {
		int parent_displacements_slot = node.layer_slots[PARENT_DISPLACEMENTS_LAYER];
		if (parent_displacements_slot >= 0) {
			vec3 ptexcoord = vec3(node.layer_origins[PARENT_DISPLACEMENTS_LAYER] + vec2((iPosition/2)*2) * node.layer_steps[PARENT_DISPLACEMENTS_LAYER], parent_displacements_slot);
			vec3 displacement = sample_displacements(ptexcoord) - node_slots[parent_displacements_slot].relative_position;
			position = mix(displacement, position, morph);
		} else {
			vec3 itexcoord =  vec3(node.layer_origins[DISPLACEMENTS_LAYER] + vec2((iPosition/2)*2) * node.layer_steps[DISPLACEMENTS_LAYER], displacements_slot);
			vec3 displacement = sample_displacements(itexcoord) - node_slots[displacements_slot].relative_position;
			position = mix(displacement, position, morph);
		}
	}

	vec3 normal = normalize(position + globals.camera);
	vec3 bitangent = normalize(cross(normal, tangents[node.face]));
	vec3 tangent = normalize(cross(normal, bitangent));

	out_position = position;
	out_texcoord = nPosition;
	out_morph = morph;
	out_normal = normal;
	out_tangent = tangent;
	out_bitangent = bitangent;
	out_i_position = vec2(iPosition);
	out_instance = gl_InstanceIndex;

	gl_Position = globals.view_proj * vec4(position, 1.0);
}
