#line 2

layout(early_fragment_tests) in;

layout(binding = 0) uniform UniformBlock {
    mat4 view_proj;
	vec3 camera;
	float padding;
} uniform_block;
layout(set = 0, binding = 1) uniform sampler linear;
layout(set = 0, binding = 2) uniform texture2DArray heights;
layout(set = 0, binding = 3) uniform texture2DArray normals;
layout(set = 0, binding = 4) uniform texture2DArray albedo;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 albedo_texcoord;
layout(location = 2) in vec3 albedo_parent_texcoord;
layout(location = 3) in vec3 normals_texcoord;
layout(location = 4) in vec3 normals_parent_texcoord;
layout(location = 5) in float morph;
layout(location = 6) in vec2 i_position;
layout(location = 7) in float side_length;
layout(location = 8) in float min_distance;
layout(location = 9) in float elevation;

layout(location = 0) out vec4 out_color;

float mipmap_level(in vec2 texture_coordinate)
{
    vec2  dx_vtc        = dFdx(texture_coordinate);
    vec2  dy_vtc        = dFdy(texture_coordinate);
    float delta_max_sqr = max(dot(dx_vtc, dx_vtc), dot(dy_vtc, dy_vtc));
    return 0.5 * log2(delta_max_sqr);
}

vec3 debug_overlay(vec3 color) {
	// if((fract(0.5*position.x/32) < 0.5) != (fract(0.5*position.z/32) < 0.5))
	// 	color = mix(color, vec3(0,0,1), 0.3);

	vec2 ts = vec2(textureSize(normals, 0).xy);
	vec2 tc = normals_texcoord.xy;//;
	float ml = mipmap_level(tc * ts);

	ml = mipmap_level(position.xz / side_length * 512);

	// if (ml < 0.0 && side_length == 128.0)
	// 	color = vec3(0.4);
	// else if (ml < -1.0)
	// 	color = vec3(1,0,0);
	// else if (ml < 0.0) // 1024
	// 	color = vec3(0.5,0,0);
	// else if (ml < 1.0) // 512
	// 	color = vec3(0,0.2,0);
	// else if (ml < 2.0) // 256
	// 	color = vec3(0,0.4,0);
	// else if (ml < 3.0) // 128
	// 	color = vec3(0,0,.7);
	// else               // 64
	// 	color = vec3(0,0,.3);
	// color = mix(color, vec3(0), 0.3-0.3*fract(ml));

	// if((fract(0.5*position.x/1024) < 0.5) != (fract(0.5*position.z/1024) < 0.5))
	// 	color = mix(color, vec3(0,0,0), 0.3);

	// if((fract(0.5*tc.x*ts.x/8) < 0.5) != (fract(0.5*tc.y*ts.y/8) < 0.5))
	// 	color = mix(color, vec3(0,0,0), 0.2);

	// if(abs(length(position.xz) - 10000.0) < 100)
	// 	color = vec3(1);
	// if(length(position.xz) < 10000) {
	// 	if((fract(0.5*position.x/1000) < 0.5) != (fract(0.5*position.z/1000) < 0.5))
	// 		color = mix(color, vec3(0,0,0), 0.3);
	// }

	// if(abs(length(position.xz-uniform_block.camera.xz) - 32*1024) < 100)
	// 	color = vec3(1);

 	// vec2 grid = abs(fract(i_position + 0.5) - 0.5) / fwidth(i_position);
	// float line = min(grid.x, grid.y);
	// color = mix(color, vec3(0.1), smoothstep(1, 0, line) * 0.6);

	// if (side_length / 512.0 <= 16.0)
	// 	color = mix(color, vec3(1,0,0), 0.4);

	// if (any(lessThan(0.5 - abs(0.5 - fract(position.xz / side_length)), vec2(0.01))))
	// 	color = mix(color, vec3(0.1), 0.3);

	// if((fract(texcoord.x*ts.x/2) < 0.5) != (fract(texcoord.y*ts.y/2) < 0.5))
	// 	color *= 0.4;

	// if(min_distance == 256.0*1.95)
	// 	color = mix(color, vec3(1,0,0), .1);
	// if(min_distance == 512.0*1.95)
	// 	color = mix(color, vec3(0,0,0), .5);
	// if(min_distance == 1024.0*1.95)
	// 	color = mix(color, vec3(0,0,1), .8);

	// color = mix(color, vec3(1,1,1), .3 * fract(heights_origin.y / 40));

 	return color;
}

vec3 extract_normal(vec2 n) {
	n = n * 2.0 - vec2(1.0);
	float y = sqrt(max(1.0 - dot(n, n),0));
	return normalize(vec3(n.x, y, n.y));
}

void main() {
	vec3 light_direction = normalize(vec3(0.4, 0.7,0.2));
	vec3 normal = extract_normal(texture(sampler2DArray(normals, linear), normals_texcoord).xy);
	if (normals_parent_texcoord.z >= 0) {
		vec3 pn = extract_normal(texture(sampler2DArray(normals, linear),
										 normals_parent_texcoord).xy);
		normal = mix(pn, normal, morph);
	}

	vec4 albedo_roughness = texture(sampler2DArray(albedo, linear), albedo_texcoord);
	if (albedo_parent_texcoord.z >= 0) {
		albedo_roughness = mix(texture(sampler2DArray(albedo, linear), albedo_parent_texcoord),
							   albedo_roughness,
							   morph);
	}

	out_color = vec4(1);
	out_color.rgb = pbr(albedo_roughness.rgb,
						albedo_roughness.a,
						position,
						normal,
						uniform_block.camera,
						vec3(0.4, 0.7,0.2),
						vec3(100000.0));

	float ev100 = 14.0;
	float exposure = 1.0 / (pow(2.0, ev100) * 1.2);
	out_color = tonemap(out_color, exposure, 2.2);

	out_color.rgb = debug_overlay(out_color.rgb);
}
