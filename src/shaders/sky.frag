#line 2

layout(set = 0, binding = 0) uniform UniformBlock {
    mat4 view_proj;
	dvec3 camera;
	double padding;
} ubo;
layout(set = 0, binding = 1) uniform sampler linear;
layout(set = 0, binding = 2) uniform texture2D sky;
layout(set = 0, binding = 3) uniform texture2D transmittance;
layout(set = 0, binding = 4) uniform texture3D inscattering;

layout(location = 0) in vec4 position;
layout(location = 1) flat in mat4 view_proj_inv;

layout(location = 0) out vec4 OutColor;

const float PI = 3.141592;
const vec3 rayleigh_Bs = vec3(5.8e-6, 13.5e-6, 33.1e-6);
const float planetRadius = 6371000.0;
const float atmosphereRadius = 6371000.0 + 100000.0;

float rayleigh_phase(float mu) {
	return 3.0 / (16.0 * PI) * (1.0 + mu * mu);
}
float mie_phase(float mu) {
	float g = 0.76;

	return 3.0 / (8.0 * PI) * ((1.0 - g * g) * (1.0 + mu * mu))
		/ ((2.0 + g * g) * pow(1.0 + g * g - 2.0 * g * mu, 1.5));
}
void reverse_parameters(float r, float mu, float mu_s,
						out float u_r, out float u_mu, out float u_mu_s) {
	float H = sqrt(atmosphereRadius * atmosphereRadius - planetRadius * planetRadius);
	float rho = sqrt(max(r * r - planetRadius * planetRadius, 0));
	float delta = r * r * mu * mu - rho * rho;

	u_r = rho / H;

	ivec3 size = textureSize(inscattering, 0);

	float hp = (size.y*0.5 - 1.0) / (size.y-1.0);
	float mu_horizon = -sqrt(1.0 - (planetRadius / r) * (planetRadius / r));
	if (mu > mu_horizon) {
		u_mu = (1.0 - hp) + hp * pow((mu - mu_horizon) / (1.0 - mu_horizon), 0.2);
	} else {
		u_mu = hp * pow((mu_horizon - mu) / (1.0 + mu_horizon), 0.2);
	}

	u_mu_s = clamp(0.5*(atan(max(mu_s, -0.45)*tan(1.26 * 0.75))
						/ 0.75 + (1.0 - 0.26)), 0, 1);
}
vec3 precomputed_atmosphere(vec3 x, vec3 x0, vec3 sun_normalized) {
	vec3 v_normalized = normalize(x0 - x);
	vec3 x_normalized = normalize(x);

	float r = clamp(length(x), planetRadius, atmosphereRadius);
	float mu = dot(v_normalized, x_normalized);
	float mu_s = dot(sun_normalized, x_normalized);
	float v = dot(v_normalized, sun_normalized);

	float u_r, u_mu, u_mu_s;
	reverse_parameters(r, mu, mu_s, u_r, u_mu, u_mu_s);

	if(u_mu <= 0.5)
		u_mu = clamp(u_mu, 0.0, 0.5 - 0.5 / textureSize(inscattering,0).y);
	else
		u_mu = clamp(u_mu, 0.5 + 0.5 / textureSize(inscattering,0).y, 1.0);

	vec4 t = texture(sampler3D(inscattering, linear), vec3(u_r, u_mu, u_mu_s));
	vec3 rayleigh = t.rgb * rayleigh_phase(v);
	vec3 mie = t.rgb * t.a / t.r * mie_phase(v) / rayleigh_Bs;
	return 10 * (rayleigh + mie);
}

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

	float lat = acos(r.z)/3.141592 * 0.5 + 0.5;
	float lon = atan(r.y, r.x) / 3.141592 * 0.5 + 0.5;
	OutColor.rgb = texture(sampler2D(sky, linear), vec2(lon, lat)).rgb * 0.025;

	vec3 x0 = r0.xyz / r0.w + vec3(ubo.camera);
	vec2 p = rsi(x0, r, atmosphereRadius);
	if (p.x < p.y && p.y > 0.0) {
		vec3 atmosphere = precomputed_atmosphere(x0 + r * max(p.x, 0.0),
												 x0 + r * p.y,
												 vec3(1, 0, 0));
		OutColor.rgb += atmosphere;
	}
}
