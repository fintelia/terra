#line 2

layout(binding = 0) uniform UniformBlock {
    mat4 view;
    mat4 projection;
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

void main() {
	vec3 position = vec3(0);
	ivec2 iPosition = ivec2((gl_VertexIndex) % (resolution+1),
							(gl_VertexIndex) / (resolution+1));

	position.xz = vec2(iPosition)
	    * (side_length / (resolution)) + in_position;
	float morph = 1 - smoothstep(0.7, 0.95, distance(position, uniform_block.camera) / min_distance);
	morph = min(morph * 2, 1);
	// if(colors_layer.y < 0)
	// 	morph = 1;

	// position.y = texture(heights, heightsOrigin + vec3(vec2(iPosition + 0.5) / textureSize(heights, 0).xy, 0)).r;

	ivec2 morphTarget = (iPosition / 2) * 2;
	float morphHeight = 0.0; // texture(heights, heightsOrigin + vec3(vec2(morphTarget + 0.5) / textureSize(heights, 0).xy, 0)).r;

	vec2 nPosition = mix(vec2(morphTarget), vec2(iPosition), morph);

	position.y = mix(morphHeight, position.y, morph);
	position.xz = nPosition * (side_length / (resolution)) + in_position;

	gl_Position = uniform_block.projection * uniform_block.view * vec4(position, 1.0);
}
