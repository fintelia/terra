#line 2
uniform mat4 modelViewProjection;

in vec3 fPosition;

out vec4 OutColor;

void main() {
	OutColor = vec4((fPosition.y), 0, 0, 1);
}
