#line 2

uniform sampler2D color;
uniform vec3 cameraPosition;
uniform vec3 sunDirection;

in vec3 fPosition;
out vec4 OutColor;

void main() {
	vec3 normal = normalize(fPosition + vec3(0,planetRadius,0));
	vec2 texcoord = 0.5 + 0.5 * vec2(acos(normal.y)) / 3.141592 * normalize(normal.xz);
	vec4 color = texture(color, texcoord);

	OutColor.rgb = mix(color.rgb, vec3(0,0.05,0.1), color.a);
	OutColor.rgb *= dot(normal, sunDirection.xyz * vec3(1,1,-1));
	OutColor.rgb = aerial_perspective(OutColor.rgb, fPosition, cameraPosition, sunDirection);
	OutColor.rgb += dither();
}
