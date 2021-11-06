
const float planetRadius = 6371000.0;
const float atmosphereRadius = 6371000.0 + 100000.0;

const vec3 rayleigh_Bs = vec3(5.8e-6, 13.5e-6, 33.1e-6);

float rayleigh_phase(float mu) {
	return 3.0 / (16.0 * 3.141592) * (1.0 + mu * mu);
	// return 0.8 * (1.4 + 0.5 * mu) / 10;
}
float mie_phase(float mu) {
	float g = 0.76;

	return 3.0 / (8.0 * 3.141592) * ((1.0 - g * g) * (1.0 + mu * mu))
		/ ((2.0 + g * g) * pow(1.0 + g * g - 2.0 * g * mu, 1.5));
}

vec3 precomputed_transmittance(float r, float mu);

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

vec3 atmosphere(vec3 r0, vec3 r1, vec3 pSun) {
	float iSun = 100000.0;
	vec3 kRlh = vec3(5.8e-6, 13.5e-6, 33.1e-6);
	float kMie = 2.0e-6;
	float shRlh = 8000.0;
	float shMie = 1200.0;
	float g = 0.76;

	const int iSteps = 32;

    // Normalize the sun and view directions.
    vec3 r = normalize(r1 - r0);

	// Primary ray initialization.
    float iStepSize = distance(r0, r1) / float(iSteps);
    float iTime = iStepSize * 0.5;

    // Initialize accumulators for Rayleigh and Mie scattering.
    vec3 totalRlh = vec3(0,0,0);
    vec3 totalMie = vec3(0,0,0);

    // Initialize optical depth accumulators for the primary ray.
    float iOdRlh = 0.0;
    float iOdMie = 0.0;

    // Sample the primary ray.
    for (int i = 0; i < iSteps; i++) {
        // Calculate the primary ray sample position.
        vec3 iPos = r0 + r * iTime;

        // Calculate the height of the sample.
        float iHeight = length(iPos) - planetRadius;

        // Calculate the optical depth of the Rayleigh and Mie scattering for this step.
        float odStepRlh = exp(-iHeight / shRlh) * iStepSize;
        float odStepMie = exp(-iHeight / shMie) * iStepSize;

        // Accumulate optical depth.
        iOdRlh += odStepRlh;
        iOdMie += odStepMie;

		// Check whether ray hits the earth
		vec2 p = rsi(iPos, pSun, planetRadius);
		if (p.x > p.y || p.y < 0) {
			float mu = dot(pSun, normalize(iPos));
			vec3 t = precomputed_transmittance(length(iPos), mu);

			// Calculate attenuation.
			vec3 attn = exp(-kRlh * iOdRlh) * exp(-kMie * iOdMie) * t;

			// Accumulate scattering.
			totalRlh += odStepRlh * attn;
			totalMie += odStepMie * attn;
		}

        // Increment the primary ray time.
        iTime += iStepSize;
    }

    // Calculate and return the final color.
    float mu = dot(r, pSun);
    return iSun * (rayleigh_phase(mu) * kRlh * totalRlh + mie_phase(mu) * kMie * totalMie);
}

// void reverse_parameters(float r, float mu, float mu_s,
// 						out float u_r, out float u_mu, out float u_mu_s) {
// 	float H = sqrt(atmosphereRadius * atmosphereRadius - planetRadius * planetRadius);
// 	float rho = sqrt(max(r * r - planetRadius * planetRadius, 0));
// 	float delta = r * r * mu * mu - rho * rho;
// 
// 	u_r = rho / H;
// 
// 	ivec3 size = textureSize(inscattering, 0);
// 
// 	float hp = (size.y*0.5 - 1.0) / (size.y-1.0);
// 	float mu_horizon = -sqrt(1.0 - (planetRadius / r) * (planetRadius / r));
// 	if (mu > mu_horizon) {
// 		u_mu = (1.0 - hp) + hp * pow((mu - mu_horizon) / (1.0 - mu_horizon), 0.2);
// 	} else {
// 		u_mu = hp * pow((mu_horizon - mu) / (1.0 + mu_horizon), 0.2);
// 	}
// 
// 	u_mu_s = clamp(0.5*(atan(max(mu_s, -0.45)*tan(1.26 * 0.75))
// 						/ 0.75 + (1.0 - 0.26)), 0, 1);
// }
// 
vec3 precomputed_transmittance(float r, float mu) {
	vec2 size = textureSize(transmittance, 0);

	float H = sqrt(atmosphereRadius * atmosphereRadius - planetRadius * planetRadius);
	float rho = sqrt(r * r - planetRadius * planetRadius);
	float u_r = clamp(rho / H, 0, 1);

	float u_mu;
	float hp = (size.y*0.5 - 1.0) / (size.y-1.0);

	float mu_horizon = -sqrt(r * r - planetRadius * planetRadius) / r;
	if (mu > mu_horizon) {
		float uu = pow((mu - mu_horizon) / (1.0 - mu_horizon), 0.2);
		u_mu = uu * hp + (1.0 - hp);
	} else {
		float uu = pow((mu_horizon - mu) / (1.0 + mu_horizon), 0.2);
		u_mu = uu * hp;
	}

	return texture(sampler2D(transmittance, nearest), (vec2(u_r, u_mu) * (size-1) + 0.5) / size).rgb;
}

vec3 precomputed_transmittance2(vec3 x, vec3 y) {
	float r1 = length(x);
	float r2 = length(y);
	float mu1 = dot(normalize(x), normalize(x - y));
	float mu2 = dot(normalize(y), normalize(x - y));

	vec2 size = textureSize(transmittance, 0);
	float hp = (size.y*0.5 - 1.0) / (size.y-1.0);

	float mu1_horizon = -sqrt(1.0 - (planetRadius / r1) * (planetRadius / r1));
	float mu2_horizon = -sqrt(1.0 - (planetRadius / r2) * (planetRadius / r2));

	float H = sqrt(atmosphereRadius * atmosphereRadius - planetRadius * planetRadius);
	float rho1 = sqrt(max(r1 * r1 - planetRadius * planetRadius, 0));
	float rho2 = sqrt(max(r2 * r2 - planetRadius * planetRadius, 0));
	float u_r1 = rho1 / H;
	float u_r2 = rho2 / H;

	float u_mu1, u_mu2;
	if (mu1 > mu1_horizon) {
		u_mu1 = (1.0 - hp) + hp * pow((mu1 - mu1_horizon) / (1.0 - mu1_horizon), 0.2);
		u_mu2 = (1.0 - hp) + hp * pow(max(mu2 - mu2_horizon, 0) / (1.0 - mu2_horizon), 0.2);
	} else {
		u_mu1 = hp * pow((mu1_horizon - mu1) / (1.0 + mu1_horizon), 0.2);
		u_mu2 = hp * pow(max(mu2_horizon - mu2, 0) / (1.0 + mu2_horizon), 0.2);
	}

	vec3 t1 = texture(sampler2D(transmittance, nearest), (vec2(u_r1, u_mu1) * (size-1) + 0.5) / size).rgb;
	vec3 t2 = texture(sampler2D(transmittance, nearest), (vec2(u_r2, u_mu2) * (size-1) + 0.5) / size).rgb;

	return t2 / t1;
}

// vec3 precomputed_atmosphere(vec3 x, vec3 x0, vec3 sun_normalized) {
// 	// if (dot(x, sun_normalized) < 0)
// 	// 	return vec3(0);
// 
// 	vec3 v_normalized = normalize(x0 - x);
// 	vec3 x_normalized = normalize(x);
// 
// 	float r = clamp(length(x), planetRadius, atmosphereRadius);
// 	float mu = dot(v_normalized, x_normalized);
// 	float mu_s = dot(sun_normalized, x_normalized);
// 	float v = dot(v_normalized, sun_normalized);
// 
// 	float u_r, u_mu, u_mu_s;
// 	reverse_parameters(r, mu, mu_s, u_r, u_mu, u_mu_s);
// 
// 	// if (mu_s < -0.45)
// 	// 	return vec3(0);
// 
// 	if(u_mu <= 0.5)
// 		u_mu = clamp(u_mu, 0.0, 0.5 - 0.5 / textureSize(inscattering,0).y);
// 	else
// 		u_mu = clamp(u_mu, 0.5 + 0.5 / textureSize(inscattering,0).y, 1.0);
// 
// 	//	return vec3(fract((u_mu - 0.5)*128));
// 
// 	vec4 t = texture(sampler3D(inscattering, linear), vec3(u_r, u_mu, u_mu_s));
// 	vec3 rayleigh = t.rgb * rayleigh_phase(v);
// 	vec3 mie = t.rgb * t.a / t.r * mie_phase(v) / rayleigh_Bs;
// 	return rayleigh + mie;
// }
// 
// vec3 precomputed_atmosphere2(vec3 x, vec3 x0, vec3 sun_normalized, out float u_mu,
// 							 bool force_miss_ground, bool force_hit_ground) {
// 	vec3 v_normalized = normalize(x0 - x);
// 	vec3 x_normalized = normalize(x);
// 
// 	float r = max(length(x), planetRadius);
// 	if(r > atmosphereRadius) {
// 		vec2 p = rsi(x, v_normalized, atmosphereRadius);
// 		if (p.x > p.y || p.y < 0.0) {
// 			return vec3(0);
// 		}
// 		x = x + v_normalized * max(p.x, 0.0);
// 		x_normalized = normalize(x);
// 		r = length(x);
// 	}
// 	float mu = dot(v_normalized, x_normalized);
// 	float mu_s = dot(sun_normalized, x_normalized);
// 	float v = dot(v_normalized, sun_normalized);
// 
// 	float u_r, u_mu_s;
// 	reverse_parameters(r, mu, mu_s, u_r, u_mu, u_mu_s);
// 
// 	if(force_hit_ground) {
// 		u_mu = min(u_mu, 0.5 - 0.5 / textureSize(inscattering,0).y);
// 	} else if(force_miss_ground) {
// 		u_mu = max(u_mu, 0.5 + 0.5 / textureSize(inscattering,0).y);
// 	}
// 
// 	if(force_miss_ground && u_mu <= 0.5) {
// 		u_mu = 1.0;
// 	} else if(force_hit_ground && u_mu >= 0.5) {
// 		u_mu = 0.0;
// 	}
// 
// 	vec4 t = texture(sampler3D(inscattering, linear), vec3(u_r, u_mu, u_mu_s));
// 	vec3 rayleigh = t.rgb*rayleigh_phase(v);
// 	vec3 mie = t.rgb * t.a / max(t.r,1e-9) * mie_phase(v);
// 
// 	return rayleigh + mie;
// }
// 
// vec3 precomputed_aerial_perspective(vec3 color, vec3 x1, vec3 x0, vec3 sun_normalized) {
// 	// vec4 hwp = worldToWarped * vec4(position, 1);
// 	// vec4 hwc = worldToWarped * vec4(cameraPosition, 1);
// 	// vec3 wp = hwp.xyz / hwp.w;
// 	// vec3 wc = hwc.xyz / hwc.w;
// 
// 	vec3 r = normalize(x1 - x0);
// 	vec2 p = rsi(x0, r, atmosphereRadius);
// 	if (p.x > p.y || p.y <= 0.0)
// 		return color;
// 
// 	vec3 wp = x1;
// 	vec3 wc = x0 + r * max(p.x, 0.0);
// 	vec3 ws = sun_normalized;
// 
// 	if (dot(sun_normalized, x1) < 0) {
// 		float d = length(cross(wp, sun_normalized));
// 
// 		vec3 u = cross(wp, sun_normalized);
// 		vec3 c = normalize(cross(sun_normalized, u));
// 
// 		wp -= r * min((planetRadius - d), distance(x0, x1));
// 
// 		// return vec3(c.xyz/10);
// 		// vec3 c = cross(x1, sun_normalized);
// 		// vec3 r = normalize(cross(sun_normalized, c)) * planetRadius;
// 		// return color;
// 	}
// 
// 	vec3 t = precomputed_transmittance2(wc, wp);
// 
// 	float u_mu;
// 	vec3 i0 = precomputed_atmosphere2(wc, wp, ws, u_mu, false, false);
// 	vec3 i1 = precomputed_atmosphere2(wp, wp + (wp - wc), ws, u_mu, u_mu > 0.5, u_mu < 0.5);
// 	vec3 inscattering = i0 - t * i1;
// 
// 	//	if(u_mu < 0.5) return vec3(0);
// 	//	return 100 * (i1 - t * i0);
// 	// if(inscattering.x <= 0 || inscattering.y <= 0 || inscattering.z <= 0)
// 	// 	return vec3(1,0,0);
// 
// 	// if(distance(wc, wp) > 100000) return vec3(1,0,0);
// 	// if(t.x > .991) return vec3(0);
// 	//	if(u_mu > 0.5) return vec3(0);
// 
// 	return  max(inscattering, vec3(0)) + color;
// }
