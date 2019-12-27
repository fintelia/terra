#line 2

layout(early_fragment_tests) in;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 1) uniform sampler linear;
layout(set = 0, binding = 2) uniform texture2DArray heights;
layout(set = 0, binding = 3) uniform texture2DArray normals;
layout(set = 0, binding = 4) uniform texture2DArray albedo;

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texcoord;
layout(location = 2) in vec2 parent_texcoord;
layout(location = 3) in vec2 albedo_layer;
layout(location = 4) in vec2 normals_layer;
layout(location = 5) in vec2 splats_layer;
layout(location = 6) in float morph;
layout(location = 7) in vec2 i_position;
layout(location = 8) in float side_length;
layout(location = 9) in float min_distance;
layout(location = 10) in vec3 camera;

float mipmap_level(in vec2 texture_coordinate)
{
    vec2  dx_vtc        = dFdx(texture_coordinate);
    vec2  dy_vtc        = dFdy(texture_coordinate);
    float delta_max_sqr = max(dot(dx_vtc, dx_vtc), dot(dy_vtc, dy_vtc));
    return 0.5 * log2(delta_max_sqr);
}

vec3 debug_overlay(vec3 color) {
	// if((fract(0.5*position.x/1024) < 0.5) != (fract(0.5*position.z/1024) < 0.5))
	// 	color = mix(color, vec3(0,0,1), 0.3);

	vec2 ts = vec2(textureSize(normals, 0).xy);
	vec2 tc = texcoord;//;
	float ml = mipmap_level(tc * ts);

	ml = mipmap_level(position.xz / side_length * 512);

	if (ml < 0.0 && side_length == 128.0)
		color = vec3(0.4);
	else if (ml < -1.0)
		color = vec3(1,0,0);
	else if (ml < 0.0)
		color = vec3(0.5,0,0);
	else if (ml < 1.0)
		color = vec3(0,0.2,0);
	else if (ml < 2.0)
		color = vec3(0,0.5,0);
	else if (ml < 3.0)
		color = vec3(0,0,0.2);
	else if (ml < 4.0)
		color = vec3(0,0,0.5);
	else if (ml < 5.0)
		color = vec3(0,0,0.7);
	else
		color = vec3(0,0,1);
	// color = mix(color, vec3(0), 0.3-0.3*fract(ml));

	if((fract(0.5*position.x/8) < 0.5) != (fract(0.5*position.z/8) < 0.5))
		color = mix(color, vec3(0,0,0), 0.3);

	// if((fract(0.5*tc.x*ts.x/8) < 0.5) != (fract(0.5*tc.y*ts.y/8) < 0.5))
	// 	color = mix(color, vec3(0,0,0), 0.2);

	// if(abs(length(position.xz) - 3000.0) < 10)
	// 	color = vec3(1);
	if(abs(length(position-camera) - 16.0) < .1)
		color = vec3(1);

	// vec2 grid = abs(fract(i_position + 0.5) - 0.5) / fwidth(i_position);
	// float line = min(grid.x, grid.y);
	// color = mix(color, vec3(0.1), smoothstep(1, 0, line));

	// if((fract(texcoord.x*ts.x/2) < 0.5) != (fract(texcoord.y*ts.y/2) < 0.5))
	// 	color *= 0.4;

	// if(min_distance == 256.0)
	// 	color = mix(color, vec3(1,0,0), .5+.0*sin(texcoord.y*100));
	// if(min_distance == 512.0)
	// 	color = mix(color, vec3(0,0,0), .5+.0*sin(texcoord.y*100));
	// if(min_distance == 1024.0)
	// 	color = mix(color, vec3(0,0,1), .5+.0*sin(texcoord.y*100));

 	return color;
}

void main() {
	vec3 light_direction = normalize(vec3(0.4, 0.7,0.2));
	vec3 normal = normalize(texture(sampler2DArray(normals, linear),
									vec3(fract(texcoord), normals_layer.x)).xyz);

	vec3 color = vec3(0.7);
	if (albedo_layer.x >= 0) {
		color = texture(sampler2DArray(albedo, linear),
						vec3(fract(texcoord), albedo_layer.x)).xyz;
	}

	float nDotL = dot(normal, light_direction);
	out_color = vec4(vec3(nDotL) * pow(color, vec3(2.2)), 1.0f);

	out_color.rgb = debug_overlay(out_color.rgb);

	// Gamma correct output color
	out_color.rgb = pow(out_color.rgb, vec3(2.2));
}
