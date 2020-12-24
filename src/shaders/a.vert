#line 2

layout(set = 0, binding = 0) uniform UniformBlock {
    mat4 view_proj;
	vec3 camera;
	float padding;
} ubo;

struct LayerDesc {
	vec3 origin;
	float _step;
	vec3 parent_origin;
	float parent_step;
};
layout(set = 0, binding = 1, std140) uniform NodeBlock {
	LayerDesc displacements;
	LayerDesc albedo;
	LayerDesc roughness;
	LayerDesc normals;
	uint resolution;
	uint face;
	uint level;
	uint padding0;
	vec3 relative_position;
	float min_distance;
	vec3 parent_relative_position;
	float padding1;
} node;

layout(set = 0, binding = 2) uniform sampler linear;
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

	vec3 position = texture(sampler2DArray(displacements, linear),
		node.displacements.origin + vec3(vec2(iPosition) * node.displacements._step, 0)).xyz - node.relative_position;

	float morph = 1 - smoothstep(0.9, 1, length(position) / node.min_distance);
	vec2 nPosition = mix(vec2((iPosition / 2) * 2), vec2(iPosition), morph);

	if (node.displacements.parent_origin.z >= 0 && morph < 1.0) {
		position = mix(
	 		texture(sampler2DArray(displacements, linear), node.displacements.parent_origin + vec3(nPosition * node.displacements.parent_step, 0)).xyz - node.parent_relative_position,
	 		texture(sampler2DArray(displacements, linear), node.displacements.origin + vec3(nPosition * node.displacements._step, 0)).xyz - node.relative_position,
			morph);
	} else {   
		position = texture(sampler2DArray(displacements, linear), node.displacements.origin + vec3(nPosition * node.displacements._step, 0)).xyz - node.relative_position;
	}

	vec3 normal = normalize(position + ubo.camera);
	vec3 bitangent = normalize(cross(normal, tangents[node.face]));
	vec3 tangent = normalize(cross(normal, bitangent));

	out_position = position;
	out_texcoord = nPosition;
	out_morph = morph;
	out_normal = normal;
	out_tangent = tangent;
	out_bitangent = bitangent;
	out_i_position = vec2(iPosition);

	gl_Position = ubo.view_proj * vec4(position, 1.0);
}
