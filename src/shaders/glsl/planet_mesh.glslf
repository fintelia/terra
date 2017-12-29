#line 2

uniform vec3 sunDirection;

in vec3 fPosition;
out vec4 OutColor;

void main() {
	vec3 normal = normalize(fPosition + vec3(0,planetRadius,0));
	OutColor = vec4(vec3(1, 0, 0) * dot(normal, sunDirection), 1);
}
