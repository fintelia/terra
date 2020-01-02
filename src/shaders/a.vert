#line 2

layout(binding = 0) uniform UniformBlock {
    mat4 view_proj;
	vec3 camera;
	float padding;
} uniform_block;

layout(location = 0) in vec2 in_position;
layout(location = 0, component=2) in float side_length;
layout(location = 0, component=3) in float min_distance;
layout(location = 1, component=0) in vec3 heights_origin;
layout(location = 1, component=3) in float heights_step;
layout(location = 2, component=0) in vec3 pheights_origin;
layout(location = 2, component=3) in float pheights_step;
layout(location = 3, component=0) in vec3 albedo_origin;
layout(location = 3, component=3) in float albedo_step;
layout(location = 4, component=0) in vec3 palbedo_origin;
layout(location = 4, component=3) in float palbedo_step;
layout(location = 5, component=0) in vec3 normals_origin;
layout(location = 5, component=3) in float normals_step;
layout(location = 6, component=0) in vec3 pnormals_origin;
layout(location = 6, component=3) in float pnormals_step;
layout(location = 7) in int resolution;

layout(set = 0, binding = 1) uniform sampler linear;
layout(set = 0, binding = 2) uniform texture2DArray heights;

layout(location = 0) out vec3 out_position;
layout(location = 1) out vec3 out_albedo_texcoord;
layout(location = 2) out vec3 out_palbedo_texcoord;
layout(location = 3) out vec3 out_normals_texcoord;
layout(location = 4) out vec3 out_pnormals_texcoord;
layout(location = 5) out float out_morph;
layout(location = 6) out vec2 out_i_position;
layout(location = 7) out float out_side_length;
layout(location = 8) out float out_min_distance;
layout(location = 9) out float out_elevation;

void main() {
	ivec2 iPosition = ivec2((gl_VertexIndex) % (resolution+1),
							(gl_VertexIndex) / (resolution+1));

	vec2 gridPosition = vec2(iPosition) * (side_length / (resolution)) + in_position;
	float morph = 1 - smoothstep(0.7, 0.95, distance(gridPosition, uniform_block.camera.xz) / min_distance);
	morph = min(morph * 2, 1);
	// if(is_top_level)
	//	morph = 1;
	vec2 nPosition = mix(vec2((iPosition / 2) * 2), vec2(iPosition), morph);


	vec3 offset = texture(sampler2DArray(heights, linear),
						  heights_origin + vec3(vec2(nPosition) * heights_step, 0)).xyz;
	if (pheights_origin.z >= 0) {
		offset = mix(texture(sampler2DArray(heights, linear),
							 pheights_origin + vec3(vec2(nPosition) * pheights_step, 0)).xyz,
					 offset,
					 morph);
	}

	vec3 position = vec3(nPosition * (side_length / resolution) + in_position, 0).xzy + offset;

	out_position = position;
	out_albedo_texcoord = albedo_origin + vec3(nPosition * albedo_step, 0);
	out_palbedo_texcoord = palbedo_origin + vec3(nPosition * palbedo_step, 0);
	out_normals_texcoord = normals_origin + vec3(nPosition * normals_step, 0);
	out_pnormals_texcoord = pnormals_origin + vec3(nPosition * pnormals_step, 0);
	out_morph = morph;
	out_i_position = vec2(iPosition);
	out_side_length = side_length;
	out_min_distance = min_distance;
	out_elevation = texture(sampler2DArray(heights, linear),
							heights_origin + vec3(nPosition * heights_step, 0)).g;

	gl_Position = uniform_block.view_proj * vec4(position, 1.0);

	// TODO: This should not be needed
	gl_Position.x *= -1;
}
