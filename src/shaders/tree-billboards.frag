#version 450 core
#include "declarations.glsl"
#include "pbr.glsl"

layout(early_fragment_tests) in;

layout(set = 0, binding = 0) uniform UniformBlock {
	Globals globals;
};
layout(set = 0, binding = 8, std140) readonly buffer Nodes {
	Node nodes[];
};

//layout(set = 0, binding = 1) uniform texture2DArray aerial_perspective;
layout(set = 0, binding = 3) uniform sampler linear;

layout(binding = 4) uniform texture2DArray billboards_albedo;
layout(binding = 5) uniform texture2DArray billboards_normals;
layout(binding = 6) uniform texture2DArray billboards_ao;
layout(binding = 7) uniform texture2DArray billboards_depth;

#ifndef SHADOWPASS
layout(binding = 9) uniform texture2D shadowmap;
layout(binding = 10) uniform samplerShadow shadow_sampler;
layout(location = 0) out vec4 out_color;
#endif

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec2 texcoord;
layout(location = 3) in vec3 normal;
layout(location = 4) flat sample in uint slot;
layout(location = 5) in vec3 right;
layout(location = 6) in vec3 up;
layout(location = 7) in vec4 atmosphere;

vec3 extract_normal(vec2 n) {
	n = n * 2.0 - vec2(1.0);
	float y = sqrt(max(1.0 - dot(n, n),0));
	return normalize(vec3(n.x, y, n.y));
}

float mip_map_level(in vec2 texture_coordinate)
{
    vec2  dx_vtc        = dFdx(texture_coordinate);
    vec2  dy_vtc        = dFdy(texture_coordinate);
    float delta_max_sqr = max(dot(dx_vtc, dx_vtc), dot(dy_vtc, dy_vtc));
    return 0.5 * log2(delta_max_sqr);
}

void main() {
	vec4 albedo = texture(sampler2DArray(billboards_albedo, linear), vec3(texcoord/6.0, 0));
	vec2 tx_normal = texture(sampler2DArray(billboards_normals, linear), vec3(texcoord/6.0, 0)).xy;
	float ao = texture(sampler2DArray(billboards_ao, linear), vec3(texcoord/6.0+1./6, 0), 0).x;
	float depth = texture(sampler2DArray(billboards_depth, linear), vec3(texcoord/6.0, 0)).x;

	//albedo.rgb *= 0.15;
	albedo.rgb = vec3(0.01,0.02,0.0);
	albedo.rgb += (color-0.5) * 0.01;

	float tx_normal_z = sqrt(max(0, 1-dot(tx_normal, tx_normal)));
	vec3 true_normal = normalize(tx_normal.x * right - tx_normal.y * up + tx_normal_z * normal);
	//normalize(position+globals.camera);//
	// true_normal.y += 1;
	// true_normal = normalize(true_normal);
	true_normal = up;

	// albedo.rgb = 0*vec3(0.013,0.037,0.0);


	if (albedo.a < 0.5)
		discard;

#ifndef SHADOWPASS

	float shadow = 0;
	// vec4 proj_position = globals.shadow_view_proj * vec4(position + normal * depth*10, 1);
	// //proj_position.xyz /= proj_position.w;
	// vec2 shadow_coord = proj_position.xy * 0.5 * vec2(1,-1) + 0.5;
	// if (all(greaterThan(shadow_coord,vec2(0))) && all(lessThan(shadow_coord,vec2(1)))) {
	// 	float depth = proj_position.z - 4.0 / 102400.0;
	// 	shadow = textureLod(sampler2DShadow(shadowmap, shadow_sampler), vec3(shadow_coord, depth), 0);
	// }

	out_color = vec4(1);
	out_color.rgb = pbr(albedo.rgb,
						0.8,
						position,
						true_normal,
						globals.camera,
						globals.sun_direction,
						vec3(100000.0)) * (1-shadow);


	// out_color.rgb += (1 - ao) * albedo.rgb * 15000 * max(0, dot(up, globals.sun_direction));// * max(dot(true_normal, up), 0);

	// vec4 ap = texture(sampler2DArray(aerial_perspective, linear), layer_to_texcoord(AERIAL_PERSPECTIVE_LAYER));
	out_color.rgb *= atmosphere.a;
	out_color.rgb += atmosphere.rgb * 16.0;


	out_color = tonemap(out_color, globals.exposure, 2.2);

	// out_color.rgb = vec3(dot(globals.sun_direction,true_normal));


	//out_color.rgb = vec3((tx_normal.x)*0.5 + 0.5);

	// out_color.rgb = vec3(1)*0.5 + 0.5;

	// if (dot(normal, up) < 0.001)
	//   	out_color.rgb = vec3(1,0,0);


	// float level = mip_map_level(texcoord*1024.0);
	// if (level < -1)  out_color.rgb = vec3(.4,0,0);
	// else if (level < 0)  out_color.rgb = vec3(1,0,0);
	// else if (level < 1)  out_color.rgb = vec3(0,1,0);
	// else if (level < 5)  out_color.rgb = vec3(0,0,1);
#endif
}
