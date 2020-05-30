#line 2

layout(early_fragment_tests) in;

layout(binding = 0) uniform UniformBlock {
	dvec4 local_origin;
    mat4 view_proj;
	vec4 camera;
} ubo;
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
layout(location = 7) in float resolution;
layout(location = 8) in float min_distance;
layout(location = 9) in float elevation;
layout(location = 10) in float face;
layout(location = 11) in float level_resolution;

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

	vec2 ts = vec2(520);//vec2(textureSize(normals, 0).xy);
	vec2 tc = normals_texcoord.xy;//;
	float ml = mipmap_level(tc * ts);

	// vec3 pc = normalize((position+vec3(0,6371000.0,0)) / (position.y+6371000.0));
	// vec3 cc = normalize((ubo.camera.xyz+vec3(0,6371000.0,0)) / (ubo.camera.y+6371000.0));
	// if(distance(pc, cc) < min_distance && distance(pc, cc) > min_distance*0.9)
	// 	color.rgb = mix(color.rgb, vec3(1,0,0), 0.3);

	// vec2 ip = abs(vec2(1) - 2*i_position/resolution);
	// if(ip.x > 0.99 || ip.y > 0.99)
	// 	color.rgb = mix(color.rgb, vec3(0,0,0), 0.8);

	// ml = mipmap_level(normals_texcoord.xy*vec2(textureSize(normals,0).xy));
	// vec3 overlay_color = vec3(0);
	// if (ml < 0.0 && side_length <= 16.0)
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
	// 	overlay_color = vec3(0,0,.3);
	// // overlay_color = mix(overlay_color, vec3(0), 0.3-0.3*fract(ml));
	// color = mix(color, overlay_color, 0.9);

	// if((fract(0.5*position.x/(4*1024*1024)) < 0.5) != (fract(0.5*position.z/(4*1024*1024)) < 0.5))
	// 	color = mix(color, vec3(0,0,0), 0.2);

	// if((fract(0.5*tc.x*ts.x/8) < 0.5) != (fract(0.5*tc.y*ts.y/8) < 0.5))
	// 	color = mix(color, vec3(0,0,0), 0.2);

	// if(length(position.xz) > 30000.0)
	// 	color = mix(color, vec3(0), 0.3);
	// if(length(position.xz) < 10000) {
	// 	if((fract(0.5*position.x/1000) < 0.5) != (fract(0.5*position.z/1000) < 0.5))
	// 		color = mix(color, vec3(0,0,0), 0.3);
	// }

	// if(abs(length(position.xz-ubo.camera.xz) - 32*1024) < 100)
	// 	color = vec3(1);

	// if(abs(max(abs(position.x), abs(position.z)) - 2048*1.5) < 30)
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

	// if(min_distance == 16*64.0*1.95)
	// 	color = mix(color, vec3(0,1,0), .1);
	// if(min_distance == 16*512.0*1.95)
	// 	color = mix(color, vec3(1,0,0), .1);
	// if(min_distance == 32*512.0*1.95)
	// 	color = mix(color, vec3(0,0,1), .1);

	// color = mix(color, vec3(1,1,1), .3 * fract(heights_origin.y / 40));
	// color = mix(color, vec3(1,1,1), .3 * fract(position.y / 100-0.5));

	// if(face == 0) color = mix(color, vec3(1,0,0), .3);
	// if(face == 1) color = mix(color, vec3(0,1,0), .3);
	// if(face == 2) color = mix(color, vec3(0,0,1), .3);
	// if(face == 3) color = mix(color, vec3(1,1,0), .3);
	// if(face == 4) color = mix(color, vec3(1,1,1), .3);
	// if(face == 5) color = mix(color, vec3(0,0,0), .3);
	// if(level_resolution == 128*1024)color = mix(color, vec3(1,0,0), .4);
	// if(level_resolution == 64*1024)color = mix(color, vec3(0,1,0), .4);
	// if(level_resolution == 32*1024)color = mix(color, vec3(0,0,1), .4);
	// if(level_resolution == 16*1024)color = mix(color, vec3(1,1,1), .4);

	// if(resolution == 64)color = mix(color, vec3(0,0,1), .4);

	// if(resolution == 32 && sin(gl_FragCoord.x) * sin(gl_FragCoord.y) < 0.4 )discard;
	// if(face == 2) discard;

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

	// if (length(position.xz-ubo.camera.xz) < 5000 && position.y < 50) {
	// 	float t = smoothstep(50,40, position.y);
	// 	albedo_roughness = mix(albedo_roughness, vec4(vec3(0.002,.007,.003), 0.1), t);
	// 	normal = mix(normal, vec3(0,1,0), t);
	// }
	// if(albedo_roughness.a == float(int(0.35*255))/255)
	// 	albedo_roughness.a = 0.7;

	out_color = vec4(1);
	out_color.rgb = pbr(albedo_roughness.rgb,
						albedo_roughness.a,
						position,
						normal,
						ubo.camera.xyz,
						vec3(0.4, 0.7, 0.2),
						vec3(100000.0));

	float ev100 = 15.0;
	float exposure = 1.0 / (pow(2.0, ev100) * 1.2);
	out_color = tonemap(out_color, exposure, 2.2);

	out_color.rgb = debug_overlay(out_color.rgb);
}
