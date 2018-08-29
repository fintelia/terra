#line 2

uniform vec3 cameraPosition;
uniform vec3 sunDirection;

uniform vec3 rayBottomLeft;
uniform vec3 rayBottomRight;
uniform vec3 rayTopLeft;
uniform vec3 rayTopRight;

in vec2 position;
out vec4 OutColor;

void main() {
	vec3 r = normalize(mix(mix(rayBottomLeft, rayBottomRight, position.x*0.5+0.5),
						   mix(rayTopLeft, rayTopRight, position.x*0.5+0.5),
						   position.y*0.5 + 0.5));

	if(acos(dot(r, sunDirection)) < 0.00872665) {
		OutColor = vec4(10,10,10,1);
		return;
	}

	vec3 wr = normalize((worldToWarped * vec4(r,0)).xyz);
	vec3 ws = normalize((worldToWarped * vec4(sunDirection,0)).xyz);
	vec4 hwc = worldToWarped * vec4(cameraPosition,1);
	vec3 wc = hwc.xyz / hwc.w;

	// Check ray atmosphere intersection points.
	vec2 p = rsi(wc, wr, atmosphereRadius);
	if (p.x > p.y || p.y < 0.0) {
		OutColor = vec4(0,0,0,1);
		return;
	}

	vec3 color = precomputed_atmosphere(wc + wr * max(p.x, 0.0),
										wc + wr * p.y,
										ws);

    OutColor = vec4(color + dither(), 1);
}
