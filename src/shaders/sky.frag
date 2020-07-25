#line 2

layout(set = 0, binding = 0) uniform UniformBlock {
    mat4 view_proj;
	dvec3 camera;
	double padding;
} ubo;
layout(set = 0, binding = 1) uniform sampler linear;
layout(set = 0, binding = 2) uniform texture2D sky;

layout(location = 0) in vec4 position;
layout(location = 1) flat in mat4 view_proj_inv;

layout(location = 0) out vec4 OutColor;

vec2 rsi(vec3 r0, vec3 rd, float sr) {
    // ray-sphere intersection that assumes
    // the sphere is centered at the origin.
    // No intersection when result.x > result.y
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
	vec4 r0 = view_proj_inv * vec4(position.xy, 1, 1);
	vec4 r1 = view_proj_inv * vec4(position.xy, 1e-9, 1);
	vec3 r = normalize(r1.xyz / r1.w - r0.xyz / r0.w);

	// vec2 p = rsi(r0.xyz / r0.w + vec3(ubo.camera), r, 6371000.0 + 100000.0);
	// if (p.x > p.y || p.y < 0.0) {
		float lat = acos(r.z)/3.141592 * 0.5 + 0.5;
		float lon = atan(r.y, r.x) / 3.141592 * 0.5 + 0.5;
		vec3 s = texture(sampler2D(sky, linear), vec2(lon, lat)).rgb * 0.1;
		OutColor = vec4(s, 1);
	// } else {
	// 	OutColor = vec4(0, 0.3, 1, 1);
	// }
}
