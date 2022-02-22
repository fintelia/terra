#version 450 core
#include "declarations.glsl"
#include "pbr.glsl"

layout(location = 0) in vec2 texcoord;
layout(location = 1) in float magnitude;
layout(location = 2) in vec4 position;

layout(location = 0) out vec4 OutColor;

layout(set = 0, binding = 0) uniform UniformBlock {
	Globals globals;
};
layout(set = 0, binding = 2) uniform sampler linear;
layout(set = 0, binding = 3) uniform sampler nearest;
layout(set = 0, binding = 4) uniform texture2D sky;
layout(set = 0, binding = 5) uniform texture2D transmittance;
layout(set = 0, binding = 6) uniform texture2D skyview;

#include "atmosphere.glsl"

const float PI = 3.1415926535;

void main() {
	vec2 v = texcoord * 2 - 1;
	float x = dot(v, v);

	float alpha = smoothstep(1, 0, x) * clamp(0, 1, exp(1-0.7*magnitude));

	// Sky calculations
	vec4 r0 = globals.view_proj_inverse * vec4(position.xy, 1, 1);
	vec4 r1 = globals.view_proj_inverse * vec4(position.xy, 1e-9, 1);
	vec3 r = normalize(r1.xyz / r1.w - r0.xyz / r0.w);
	vec3 camera = normalize(globals.camera);
    vec3 sun = normalize(globals.sun_direction);
    vec3 a = normalize(cross(camera, sun));
    vec3 b = normalize(cross(camera, a));
	float theta = asin(dot(r, camera));
	float phi = atan(dot(r, b), dot(r, a)) / PI * 0.5 + 0.5;
	float camera_distance = length(globals.camera);
	float min_theta = -PI/2 + asin(planetRadius / camera_distance);
    float max_theta = camera_distance < atmosphereRadius ? PI/2 : -PI/2 + asin(atmosphereRadius / camera_distance);
	float u = (theta - min_theta) / (max_theta - min_theta);
	u = sqrt(u);
	vec4 sv = texture(sampler2D(skyview, linear), (vec2(u, phi) * 127 + 0.5) / 128);

	// Non-physical transformation to make sure stars aren't visible from the ground.
	// alpha *= pow(sv.a * 16, 100);

	OutColor = vec4(vec3(1), alpha);
	// if (alpha <= 0)
	// 	OutColor = vec4(vec3(1,0,0), 1);
	// else
	// 	OutColor = vec4(1);
}
