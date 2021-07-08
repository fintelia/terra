#version 450 core
#include "declarations.glsl"
#include "pbr.glsl"

layout(set = 0, binding = 0) uniform UniformBlock {
	Globals globals;
};
layout(set = 0, binding = 1) uniform sampler linear;
layout(set = 0, binding = 2) uniform sampler nearest;
layout(set = 0, binding = 3) uniform texture2D sky;
layout(set = 0, binding = 4) uniform texture2D transmittance;

layout(location = 0) in vec4 position;

layout(location = 0) out vec4 OutColor;

const float planetRadius = 6371000.0;
const float atmosphereRadius = 6371000.0 + 100000.0;

vec2 rsi(vec3 r0, vec3 rd, float sr);
vec3 precomputed_transmittance(float r, float mu);
vec3 precomputed_atmosphere(vec3 x, vec3 x0, vec3 sun_normalized);
vec3 atmosphere(vec3 r0, vec3 r1, vec3 pSun);

void main() {
	vec4 r0 = globals.view_proj_inverse * vec4(position.xy, 1, 1);
	vec4 r1 = globals.view_proj_inverse * vec4(position.xy, 1e-9, 1);
	vec3 r = normalize(r1.xyz / r1.w - r0.xyz / r0.w);

	float lat = r.z * 0.5 + 0.5;// acos(r.z)/3.141592 * 0.5 + 0.5;
	float lon = atan(r.y, r.x) / 3.141592 * 0.5 + 0.5;
	OutColor.rgb = pow(texture(sampler2D(sky, linear), vec2(lon, lat)).rgb, vec3(5)) * 10000;

	vec3 x0 = r0.xyz / r0.w + globals.camera;
	vec2 p = rsi(x0, r, atmosphereRadius);

	if (p.x < p.y && p.y > 0.0) {
		vec3 x1 = x0 + r * p.y;
		x0 = x0 + r * max(p.x, 0.0);

		OutColor.rgb = atmosphere(x0, x1, vec3(0.4, 0.7, 0.2))
			+ OutColor.rgb * precomputed_transmittance(length(x0), dot(normalize(x0), r));
	}

	float ev100 = 15.0;
	float exposure = 1.0 / (pow(2.0, ev100) * 1.2);
	OutColor = tonemap(OutColor, exposure, 2.2);
	// if (dot(x0 + r * max(p.x, 0.0), vec3(0.4, 0.7, 0.2)) < 0)
	// 	OutColor.rgb = vec3(1,0,0);
}

#include "atmosphere.glsl"
