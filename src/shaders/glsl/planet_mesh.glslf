#line 2

uniform sampler2D color;
uniform vec3 cameraPosition;
uniform vec3 sunDirection;

in vec3 fPosition;
out vec4 OutColor;

vec3 water_color(vec3 normal) {
	vec3 ray = normalize(fPosition - cameraPosition);
	vec3 reflected = reflect(ray, normalize(normal));

	vec3 reflectedColor = vec3(0,0.1,0.2) * 0.27;//textureLod(sky, normalize(reflected), 4).rgb;
	vec3 refractedColor = vec3(0,0.1,0.2) * 0.25;

	float R0 = pow(0.333 / 2.333, 2);
	float R = R0 + (1 - R0) * pow(1 - reflected.y, 5);
	return mix(refractedColor, reflectedColor, R);
}

void main() {
	vec3 normal = normalize(fPosition + vec3(0,planetRadius,0));
	vec2 texcoord = 0.5 + 0.5 * vec2(acos(normal.y)) / 3.141592 * normalize(normal.xz);
	vec4 color = texture(color, texcoord);

	OutColor = vec4(mix(color.rgb, water_color(normal), color.a), 1);
	OutColor.rgb *= max(dot(normal, sunDirection.xyz), 0.02);
	OutColor.rgb = precomputed_aerial_perspective(OutColor.rgb, fPosition, cameraPosition, sunDirection);
	OutColor.rgb += dither();
}
