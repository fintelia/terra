#version 450 core
#include "declarations.glsl"

layout(set = 0, binding = 0, std140) uniform UniformBlock {
    Globals globals;
};

layout(set = 0, binding = 1) uniform sampler linear;
layout(set = 0, binding = 2) uniform sampler nearest;
layout(set = 0, binding = 3) uniform texture2D cloud_cover;
layout(set = 0, binding = 4) uniform texture3D cloud_shape;

layout(location = 0) in vec4 position;
layout(location = 0) out vec4 OutColor;

const float upperCloudRadius = 6371000.0 + 5000.0;
const float lowerCloudRadius = 6371000.0 + 2000.0;

// Ray-sphere intersection that assumes the sphere is centered at the origin.
// No intersection when result.x > result.y
vec2 rsi(vec3 r0, vec3 rd, float sr) {
    float a = dot(rd, rd);
    float b = 2.0 * dot(rd, r0);
    float c = dot(r0, r0) - (sr * sr);
    float d = (b*b) - 4.0*a*c;
    if (d < 0.0) return vec2(1e5,-1e5);
    return vec2(
        (-b - sqrt(d))/(2.0*a),
        (-b + sqrt(d))/(2.0*a)
    );
}

void main() {
	vec4 r0 = globals.view_proj_inverse * vec4(position.xy, 1, 1);
	vec4 r1 = globals.view_proj_inverse * vec4(position.xy, 1e-9, 1);
	vec3 r = normalize(r1.xyz / r1.w - r0.xyz / r0.w);

	vec3 x0 = r0.xyz / r0.w + globals.camera;
	vec2 p = rsi(x0, r, upperCloudRadius);

	vec3 x1 = x0 + r * p.y;
	x0 = x0 + r * max(p.x, 0.0);

	vec3 v = normalize(x0);
	vec2 polar = vec2(atan(v.y, v.x) * 0.5 / 3.141592 + 0.5, acos(v.z) / 3.141592);
	float cover = texture(sampler2D(cloud_cover, linear), polar).z;

	OutColor = vec4(vec3(0.3), 0);
	if (p.x < p.y && p.y > 0 && p.x > 0) {
		//OutColor.rgb = vec3(cover);
		OutColor.a = cover;
	}
}
