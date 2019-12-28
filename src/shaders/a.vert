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

layout(location = 4) in vec2 texture_origin;
layout(location = 5) in vec2 parent_texture_origin;
layout(location = 6) in vec2 albedo_layer;
layout(location = 7) in vec2 normals_layer;
layout(location = 8) in vec2 splats_layer;
layout(location = 9) in float texture_step;
layout(location = 10) in float parent_texture_step;
layout(location = 11) in int resolution;

layout(set = 0, binding = 1) uniform sampler linear;
layout(set = 0, binding = 2) uniform texture2DArray heights;

layout(location = 0) out vec3 out_position;
layout(location = 1) out vec3 out_albedo_texcoord;
layout(location = 2) out vec3 out_albedo_parent_texcoord;
layout(location = 3) out vec3 out_normals_texcoord;
layout(location = 4) out vec3 out_normals_parent_texcoord;
layout(location = 5) out float out_morph;
layout(location = 6) out vec2 out_i_position;
layout(location = 7) out float out_side_length;
layout(location = 8) out float out_min_distance;
layout(location = 9) out vec3 out_camera;

void main() {
	vec3 position = vec3(0);
	ivec2 iPosition = ivec2((gl_VertexIndex) % (resolution+1),
							(gl_VertexIndex) / (resolution+1));

	position.xz = vec2(iPosition)
	    * (side_length / (resolution)) + in_position;
	float morph = 1 - smoothstep(0.7, 0.95, distance(position.xz, uniform_block.camera.xz) / min_distance);
	morph = min(morph * 2, 1);
	if(albedo_layer.y < 0)
		morph = 1;

	position.y = texture(sampler2DArray(heights, linear),
						 heights_origin + vec3(vec2(iPosition * heights_step + 0.5)
											  / textureSize(heights, 0).xy, 0)).r;

	ivec2 morphTarget = (iPosition / 2) * 2;
	float morphHeight = texture(sampler2DArray(heights, linear),
								heights_origin + vec3(vec2(morphTarget * heights_step + 0.5)
													 / textureSize(heights, 0).xy, 0)).r;

	vec2 nPosition = mix(vec2(morphTarget), vec2(iPosition), morph);

	position.y = mix(morphHeight, position.y, morph);
	position.xz = nPosition * (side_length / (resolution)) + in_position;

	out_position = position;
	out_albedo_texcoord =
		vec3(texture_origin + nPosition * texture_step, albedo_layer.x);
	out_albedo_parent_texcoord =
		vec3(parent_texture_origin + nPosition * parent_texture_step, albedo_layer.y);
	out_normals_texcoord = vec3(out_albedo_texcoord.xy, normals_layer.x);
	out_normals_parent_texcoord = vec3(out_albedo_parent_texcoord.xy, normals_layer.y);
	out_morph = morph;
	out_i_position = vec2(iPosition);
	out_side_length = side_length;
	out_min_distance = min_distance;
	out_camera = uniform_block.camera;

	gl_Position = uniform_block.view_proj * vec4(position, 1.0);
}
