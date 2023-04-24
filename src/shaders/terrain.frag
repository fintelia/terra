#version 450 core
#include "declarations.glsl"
#include "pbr.glsl"

layout(early_fragment_tests) in;

layout(set = 0, binding = 0, std140) uniform UniformBlock {
    Globals globals;
};
layout(set = 0, binding = 1, std140) readonly buffer Nodes {
	Node nodes[];
};
layout(set = 0, binding = 2) uniform sampler linear;
//layout(set = 0, binding = 3) uniform sampler nearest;
layout(set = 0, binding = 3) uniform texture2DArray normals;
layout(set = 0, binding = 4) uniform texture2DArray albedo;
layout(set = 0, binding = 6) uniform texture2DArray grass_canopy;
layout(set = 0, binding = 7) uniform texture2DArray root_aerial_perspective;
//layout(set = 0, binding = 8) uniform texture2DArray displacements;
layout(set = 0, binding = 9) uniform texture2DArray aerial_perspective;
layout(set = 0, binding = 10) uniform sampler nearest;
layout(set = 0, binding = 11) uniform texture2DArray bent_normals;
// layout(set = 0, binding = 12) uniform texture2D shadowmap;
// layout(set = 0, binding = 13) uniform samplerShadow shadow_sampler;

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texcoord;
layout(location = 2) in float morph;
layout(location = 3) in vec3 normal;
layout(location = 4) in vec3 tangent;
layout(location = 5) in vec3 bitangent;
layout(location = 6) in vec2 i_position;
layout(location = 7) flat in uint instance;

layout(location = 0) out vec4 out_color;

// float mipmap_level(in vec2 texture_coordinate)
// {
//     vec2  dx_vtc        = dFdx(texture_coordinate);
//     vec2  dy_vtc        = dFdy(texture_coordinate);
//     float delta_max_sqr = max(dot(dx_vtc, dx_vtc), dot(dy_vtc, dy_vtc));
//     return 0.5 * log2(delta_max_sqr);
// }

vec3 debug_overlay(vec3 color) {
	Node node = nodes[instance];

	// if((fract(0.5*position.x/32) < 0.5) != (fract(0.5*position.z/32) < 0.5))
	// 	color = mix(color, vec3(0,0,1), 0.3);

	// vec2 ts = vec2(520);//vec2(textureSize(normals, 0).xy);
	// vec2 tc = normals_texcoord.xy;//;
	// float ml = mipmap_level(tc * ts);

	// vec3 pc = normalize((position+vec3(0,6371000.0,0)) / (position.y+6371000.0));
	// vec3 cc = normalize((globals.camera.xyz+vec3(0,6371000.0,0)) / (globals.camera.y+6371000.0));
	// if(distance(pc, cc) < min_distance && distance(pc, cc) > min_distance*0.9)
	// 	color.rgb = mix(color.rgb, vec3(1,0,0), 0.3);

	// ml = mipmap_level(normals_texcoord.xy*vec2(textureSize(normals,0).xy));
	// vec3 overlay_color = vec3(0);
	// if (ml < 0.0 /*&& side_length <= 16.0*/)
	// 	overlay_color = vec3(0.4);
	// else if (ml < -1.0)
	// 	overlay_color = vec3(1,0,0);
	// else if (ml < 0.0) // 1024
	// 	overlay_color = vec3(0.5,0,0);
	// else if (ml < 1.0) // 512
	// 	overlay_color = vec3(0,0.2,0);
	// else if (ml < 2.0) // 256
	// 	overlay_color = vec3(0,0.4,0);
	// else if (ml < 3.0) // 128
	// 	overlay_color = vec3(0,0,.7);
	// else               // 64
	// 	overlay_color = vec3(0,0,.1);
	// // overlay_color = mix(overlay_color, vec3(0), 0.3-0.3*fract(ml));
	// color = mix(color, overlay_color, 0.9);

	// if((fract(0.5*position.x/(4*1024*1024)) < 0.5) != (fract(0.5*position.z/(4*1024*1024)) < 0.5))
	// 	color = mix(color, vec3(0,0,0), 0.2);

	// if((fract(0.5*tc.x*ts.x/8) < 0.5) != (fract(0.5*tc.y*ts.y/8) < 0.5))
	// 	color = mix(color, vec3(0,0,0), 0.2);

	// if(length(position.xz) < 10000) {
	// 	if((fract(0.5*position.x/1000) < 0.5) != (fract(0.5*position.z/1000) < 0.5))
	// 		color = mix(color, vec3(0,0,0), 0.3);
	// }

	// if(abs(length(position.xz-globals.camera.xz) - 32*1024) < 100)
	// 	color = vec3(1);

	// vec3 r = normalize(position.xyz + globals.camera.xyz);
	// float latitude = acos(r.z) / 3.141592 * 180;
	// if (fract(latitude/10+.7) < 0.01)
	// 	color = vec3(1);

	// if(fract(length(position)/1024) < 0.05)
	// 	color = vec3(1);

 	// vec2 grid = abs(fract(i_position + 0.5) - 0.5) / fwidth(i_position);
	// float line = min(grid.x, grid.y);
	// color = mix(color, vec3(.5), 1.3* smoothstep(1, 0, line));

	// if (any(lessThan(0.5 - abs(0.5 - fract(position.xz / side_length)), vec2(0.01))))
	// 	color = mix(color, vec3(0.1), 0.3);

	// if((fract(texcoord.x*ts.x/2) < 0.5) != (fract(texcoord.y*ts.y/2) < 0.5))
	// 	color *= 0.4;

	// if(min_distance == 16*64.0*1.95)
	// 	color = mix(color, vec3(0,1,0), .1);
	// if(min_distance == 16*512.0*1.95)
	// 	color = mix(color, vec3(1,0,0), .1);
	// if(min_distance == 32*512.0*1.95)
	// 	color = mix(color, vec3(0,0,1), .1);

	// color = mix(color, vec3(1,1,1), .3 * fract(heights_origin.y / 40));
	// color = mix(color, vec3(1,1,1), .3 * fract(position.y / 100-0.5));

	// vec2 v = vec2(i_position) / float(level_resolution) *2;
	// // v = v * (1.4511 + (1 - 1.4511)*abs(v));
	// // v = v * (1.4511 + (1 - 1.4511)*abs(v));
	// // v = sign(v) * (1.4511 - sqrt(1.4511 * 1.4511 - 1.8044 * abs(v))) / 0.9022;
	// // v = sign(v) * (1.4511 - sqrt(1.4511 * 1.4511 - 1.8044 * abs(v))) / 0.9022;

	// float s = 0;
	// vec2 vv = fract((v*.25+0.25)*16);
	// if (vv.x < 0.5 != vv.y < 0.5) {
	// 	s += 0.1;
	// }
	// vv = fract((v*.25+0.25)*16*16);
	// if (vv.x < 0.5 != vv.y < 0.5) {
	// 	s += 0.05;
	// }
	// vv = fract((v*.25+0.25)*16*16*16);
	// if (vv.x < 0.5 != vv.y < 0.5) {
	// 	s += 0.05;
	// }

	// if(node.face == 0) color = mix(color, vec3(1,0,0), 0.1);
	// if(node.face == 1) color = mix(color, vec3(0,1,0), 0.1);
	// if(node.face == 2) color = mix(color, vec3(0,0,1), 0.1);
	// if(node.face == 3) color = mix(color, vec3(1,1,0), 0.1);
	// if(node.face == 4) color = mix(color, vec3(1,1,1), 0.1);
	// if(node.face == 5) color = mix(color, vec3(0,0,0), 0.1);

	// if(node.level == 0)color = mix(color, vec3(1,0,0), .3); // 20km
	// if(node.level == 1)color = mix(color, vec3(0,1,0), .3); // 10km
	// if(node.level == 2)color = mix(color, vec3(0,0,1), .3); //  5km
	// if(node.level == 3)color = mix(color, vec3(0,1,1), .3); //  2km
	// if(node.level == 4)color = mix(color, vec3(1,1,0), .3); //  1km
	// if(node.level == 5)color = mix(color, vec3(1,0,1), .3); // 600m
	// if(node.level == 6)color = mix(color, vec3(1,1,1), .3); // 300m
	// if(node.level == 7)color = mix(color, vec3(1,0,0), .2); // 150m
	// if(node.level == 8)color = mix(color, vec3(0,1,0), .2); //  76m
	// if(node.level == 9)color = mix(color, vec3(0,0,1), .2); //  38m
	// if(node.level == 10)color = mix(color, vec3(1,0,0), .3); // 19m cell
	// if(node.level == 11)color = mix(color, vec3(0,1,0), .3); // 10m
	// if(node.level == 12)color = mix(color, vec3(0,0,1), .3); //  5m
	// if(node.level == 13)color = mix(color, vec3(0,1,1), .3); //  2m
	// if(node.level == 14)color = mix(color, vec3(1,1,0), .3); //  1m
	// if(node.level == 15)color = mix(color, vec3(1,0,1), .3); // 60cm
	// if(node.level == 16)color = mix(color, vec3(1,1,1), .3); // 30cm
	// if(node.level == 17)color = mix(color, vec3(1,0,0), .2); // 15cm
	// if(node.level == 18)color = mix(color, vec3(0,1,0), .2); //  7cm
	// if(node.level == 19)color = mix(color, vec3(0,0,1), .2); //  4cm / 30cm vertex / 19m tile
	// if(node.level == 20)color = mix(color, vec3(0,0,0), .2); //  2cm / 30cm vertex / 10m tile


	// if (int(level_resolution/2 + i_position.x / 64) % 2 !=
	// 	int(level_resolution/2 + i_position.y / 64) % 2)
	// 	color *= 0.5;

	// if(resolution == 64)color = mix(color, vec3(0,0,1), .4);

	// if(resolution == 32 && sin(gl_FragCoord.x) * sin(gl_FragCoord.y) < 0.4 )discard;
	// if(face == 2) discard;

	// if (level_resolution > 512*pow(2,4) || level_resolution == 512*pow(2,4) && morph > 0.99)
	// 	color = mix(color, vec3(0,0,1), .4);
	// if (abs(max(max(position.x, position.y), position.z)-1050) <10)
	// if (abs(length(position)-1000) < 10)
	//  	color = mix(color, vec3(0), 0.5);

	// float line2 = abs(length(position) - 10) / fwidth(length(position)) * 0.5;
	// color = mix(color, vec3(0.1), smoothstep(1, 0, line2));

	// vec3 p = normalize(position + vec3(globals.camera));
	// vec2 coord = vec2((acos(p.z) * 180.0 / 3.141592),
	// 				  (atan(p.x, p.y)*180.0/3.141592));
	// vec2 grid = .5*fract(coord) / fwidth(fract(coord));
	// float line = min(grid.x, grid.y);
	// if (coord.x > 90-60 && coord.x < 90+56)
	// color = mix(color, vec3(0.), smoothstep(1, 0, line) * 0.6);

	// if (face != 2) discard;

	// if (node.resolution == 32)
	// 	color *= 0.5;
	// color = mix(color, vec3(0,0,0), (1-morph)*.4);

	// if(abs(length(position.xz)-32) < 2)
	// 	color = mix(color, vec3(0), 1);

 	return color;
}

vec3 extract_normal(vec2 n) {
	n = n * 2.0 - vec2(1.0);
	float y = sqrt(max(1.0 - dot(n, n),0));
	return normalize(vec3(n.x, y, n.y));
}

vec3 layer_to_texcoord(uint layer) {
	Node node = nodes[instance];
	return layer_texcoord(node.layers[layer], texcoord);
}

void main() {
	Node node = nodes[instance];

	vec3 tex_normal = extract_normal(texture(sampler2DArray(normals, linear), layer_to_texcoord(NORMALS_LAYER)).xy);
	if (node.layers[PARENT_NORMALS_LAYER].slot >= 0) {
		vec3 pn = extract_normal(textureLod(sampler2DArray(normals, linear), layer_to_texcoord(PARENT_NORMALS_LAYER), 0).xy);
		tex_normal = mix(pn, tex_normal, morph);
	}
	vec3 bent_normal = mat3(tangent, normal, bitangent) * tex_normal;

	vec4 albedo_roughness = texture(sampler2DArray(albedo, linear), layer_to_texcoord(ALBEDO_LAYER));
	if (node.layers[PARENT_ALBEDO_LAYER].slot >= 0) {
		vec4 parent_albedo_roughness = textureLod(sampler2DArray(albedo, linear), layer_to_texcoord(PARENT_ALBEDO_LAYER), 0);
		albedo_roughness = mix(parent_albedo_roughness, albedo_roughness, morph);
	}

	vec4 bn_value = texture(sampler2DArray(bent_normals, linear), layer_to_texcoord(BENT_NORMALS_LAYER));

	// if (node.grass_canopy_origin.z >= 0) {
	// 	vec4 canopy = texture(sampler2DArray(grass_canopy, linear), node.grass_canopy_origin + vec3(texcoord * node.grass_canopy_step, 0));
	// 	canopy.a *= smoothstep(512*2, 512*1, length(position));
	// 	if (length(position) < 512*2) {
	// 		albedo_value.rgb = mix(albedo_value.rgb, albedo_value.rgb + (canopy.rgb - 0.5) * 0.15, canopy.a);
	// 	}
	// }

	float shadow = 0;
	// vec4 proj_position = globals.shadow_view_proj * vec4(position, 1);
	// //proj_position.xyz /= proj_position.w;
	// vec2 shadow_coord = proj_position.xy * 0.5 * vec2(1,-1) + 0.5;
	// if (all(greaterThan(shadow_coord,vec2(0))) && all(lessThan(shadow_coord,vec2(1)))) {
	// 	float depth = proj_position.z - 4.0 / 102400.0;
	// 	shadow = textureLod(sampler2DShadow(shadowmap, shadow_sampler), vec3(shadow_coord, depth), 0);
	// }

	//bent_normal = normalize(position + globals.camera);

	out_color = vec4(1);
	out_color.rgb = pbr(albedo_roughness.rgb,
						albedo_roughness.a,
						position,
						bent_normal,
						globals.camera,
						globals.sun_direction,
						vec3(100000.0)) * (1-shadow);

	// float ambient_strength = max(0, dot(normal, globals.sun_direction)) * max(0, tex_normal.y);
	// if (node.layers[BENT_NORMALS_LAYER].slot >= 0)
	// 	out_color.rgb += bn_value.a * 15000 * albedo_roughness.rgb * ambient_strength;
	// else
	// 	out_color.rgb += 15000 * albedo_roughness.rgb * ambient_strength;

	vec4 ap;
	if (node.layers[AERIAL_PERSPECTIVE_LAYER].slot >= 0) {
		ap = textureLod(sampler2DArray(aerial_perspective, linear), layer_to_texcoord(AERIAL_PERSPECTIVE_LAYER), 0);
	} else {
		ap = textureLod(sampler2DArray(root_aerial_perspective, linear), layer_to_texcoord(ROOT_AERIAL_PERSPECTIVE_LAYER), 0);
	}
	out_color.rgb *= ap.a;
	out_color.rgb += ap.rgb * 16.0;

	out_color = tonemap(out_color, globals.exposure, 2.2);

	out_color.rgb = debug_overlay(out_color.rgb);
}
