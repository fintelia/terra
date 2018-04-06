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

	// Check ray atmosphere intersection points.
	vec2 p = rsi(cameraPosition+vec3(0,planetRadius,0), r, atmosphereRadius);
	if (p.x > p.y || p.y < 0.0) {
		OutColor = vec4(0,0,0,1);
		return;
	}

	vec3 color = precomputed_atmosphere(cameraPosition + r * max(p.x, 0.0),
										cameraPosition + r * p.y,
										sunDirection);

    // Apply exposure.
    OutColor = vec4(1.0 - exp(-0.75 * color), 1);
	OutColor.rgb += dither();
}
