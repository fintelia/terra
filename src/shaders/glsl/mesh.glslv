#line 2

uniform mat4 modelViewProjection;

in vec3 mposition;
in vec3 position;
in vec3 color;
in float rotation;
in float texture_layer;

out vec3 fPosition;

void main() {
	fPosition = position + mposition;
	gl_Position = modelViewProjection * vec4(fPosition, 1.0);
}
