#line 2

uniform mat4 modelViewProjection;
in vec3 vPosition;

void main() {
	 vec3 position = vPosition;
	 position.xz *= 1;
	gl_Position = modelViewProjection * vec4(position, 1.0);
}
