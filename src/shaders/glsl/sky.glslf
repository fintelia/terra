uniform mat4 invModelViewProjection;
uniform vec3 cameraPosition;
uniform vec3 sunDirection;

in vec2 position;

out vec4 OutColor;

void main() {
	vec4 hr0 = invModelViewProjection * vec4(position, 0, 1);
	vec4 hr1 = invModelViewProjection * vec4(position, 1, 1);
	vec3 ray = hr1.xyz / hr1.w - hr0.xyz / hr0.w;
	vec3 r = normalize(ray);

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
