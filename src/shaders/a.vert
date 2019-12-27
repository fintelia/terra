#line 2

layout(binding = 0) uniform UniformBlock {
    mat4 view_proj;
	vec3 camera;
	float padding;
} uniform_block;

layout(location = 0) in vec2 in_position;
layout(location = 1) in float side_length;
layout(location = 2) in float min_distance;
layout(location = 3) in vec3 heights_origin;
layout(location = 4) in vec2 texture_origin;
layout(location = 5) in vec2 parent_texture_origin;
layout(location = 6) in vec2 colors_layer;
layout(location = 7) in vec2 normals_layer;
layout(location = 8) in vec2 splats_layer;
layout(location = 9) in float texture_step;
layout(location = 10) in float parent_texture_step;
layout(location = 11) in int resolution;

layout(set = 0, binding = 1) uniform sampler linear;
layout(set = 0, binding = 2) uniform texture2DArray heights;

layout(location = 0) out vec3 out_position;
layout(location = 1) out vec2 out_texcoord;
layout(location = 2) out vec2 out_parent_texcoord;
layout(location = 3) out vec2 out_colors_layer;
layout(location = 4) out vec2 out_normals_layer;
layout(location = 5) out vec2 out_splats_layer;
layout(location = 6) out float out_morph;
layout(location = 7) out vec2 out_i_position;
layout(location = 8) out float out_side_length;
layout(location = 9) out float out_min_distance;
layout(location = 10) out vec3 out_camera;

void main() {
	vec3 position = vec3(0);
	ivec2 iPosition = ivec2((gl_VertexIndex) % (resolution+1),
							(gl_VertexIndex) / (resolution+1));

	position.xz = vec2(iPosition)
	    * (side_length / (resolution)) + in_position;
	float morph = 1 - smoothstep(0.7, 0.95, distance(position, uniform_block.camera) / min_distance);
	morph = min(morph * 2, 1);
	if(colors_layer.y < 0)
		morph = 1;

	position.y = texture(sampler2DArray(heights, linear),
						 heights_origin + vec3(vec2(iPosition + 0.5)
											  / textureSize(heights, 0).xy, 0)).r;

	ivec2 morphTarget = (iPosition / 2) * 2;
	float morphHeight = texture(sampler2DArray(heights, linear),
								heights_origin + vec3(vec2(morphTarget + 0.5)
													 / textureSize(heights, 0).xy, 0)).r;

	vec2 nPosition = mix(vec2(morphTarget), vec2(iPosition), morph);

	position.y = mix(morphHeight, position.y, morph);
	position.xz = nPosition * (side_length / (resolution)) + in_position;

	out_position = position;
	out_texcoord = texture_origin + nPosition * texture_step;
	out_parent_texcoord = parent_texture_origin + nPosition * parent_texture_step;
	out_colors_layer = colors_layer;
	out_normals_layer = normals_layer;
	out_splats_layer = splats_layer;
	out_morph = morph;
	out_i_position = vec2(iPosition);
	out_side_length = side_length;
	out_min_distance = min_distance;
	out_camera = uniform_block.camera;

	gl_Position = uniform_block.view_proj * vec4(position, 1.0);
}
