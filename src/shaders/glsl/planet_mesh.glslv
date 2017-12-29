#line 2

uniform mat4 modelViewProjection;

in vec3 vPosition;
out vec3 fPosition;

void main() {
	fPosition = vPosition;
	gl_Position = modelViewProjection * vec4(fPosition, 1.0);
}
