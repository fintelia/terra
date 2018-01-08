
uniform vec3 cameraPosition;
uniform vec3 sunDirection;

in vec3 ray;

out vec4 OutColor;

void main() {
	vec3 r = normalize(ray);

	// Check ray atmosphere intersection points.
	vec2 p = rsi(cameraPosition+vec3(0,planetRadius,0), r, atmosphereRadius);
	if (p.x > p.y || p.y < 0.0) {
		OutColor = vec4(0,0,0,1);
		return;
	}

	vec3 color = atmosphere(cameraPosition, cameraPosition + r * p.y, sunDirection);

    // Apply exposure.
    OutColor = vec4(1.0 - exp(-0.75 * color), 1);
}
