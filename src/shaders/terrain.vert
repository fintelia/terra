#line 2

layout(set = 0, binding = 0, std140) uniform UniformBlock {
    Globals globals;
};
layout(set = 0, binding = 1, std140) uniform NodeBlock {
	NodeState node;
};
layout(set = 0, binding = 3) uniform sampler nearest;
layout(set = 0, binding = 9) uniform texture2DArray displacements;

layout(location = 0) out vec3 out_position;
layout(location = 1) out vec2 out_texcoord;
layout(location = 2) out float out_morph;
layout(location = 3) out vec3 out_normal;
layout(location = 4) out vec3 out_tangent;
layout(location = 5) out vec3 out_bitangent;
layout(location = 6) out vec2 out_i_position;

const vec3 tangents[6] = vec3[6](
	vec3(0,1,0),
	vec3(0,-1,0),
	vec3(1,0,0),
	vec3(-1,0,0),
	vec3(1,0,0),
	vec3(-1,0,0)
);

void main() {
	ivec2 iPosition = ivec2((gl_VertexIndex) % (node.resolution+1),
							(gl_VertexIndex) / (node.resolution+1));

	vec3 texcoord = node.displacements.origin + vec3(vec2(iPosition) * node.displacements._step, 0);
	vec3 position = texture(sampler2DArray(displacements, nearest), texcoord).rgb - node.relative_position;
	
	float morph = 1 - smoothstep(0.9, 1, length(position) / node.min_distance);
	vec2 nPosition = mix(vec2((iPosition / 2) * 2), vec2(iPosition), morph);

	if (morph < 1.0) {
		if (node.displacements.parent_origin.z >= 0 && morph < 1.0) {
			vec3 ptexcoord = node.displacements.parent_origin + vec3(vec2((iPosition / 2) * 2) * node.displacements.parent_step, 0);
			vec3 displacement = texture(sampler2DArray(displacements, nearest), ptexcoord).rgb - node.parent_relative_position;
			position = mix(displacement, position, morph);
		} else {
			vec3 itexcoord = node.displacements.origin + vec3(vec2((iPosition / 2) * 2) * node.displacements._step, 0);
			vec3 displacement = texture(sampler2DArray(displacements, nearest), itexcoord).rgb - node.relative_position;
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

	gl_Position = globals.view_proj * vec4(position, 1.0);
}
