#line 2
uniform mat4 modelViewProjection;

uniform sampler2DArray normals;

in vec3 fPosition;
in vec3 fNormalsTexcoord;

out vec4 OutColor;

void main() {
	OutColor = vec4(texture(normals, fNormalsTexcoord).rgb, 1);
}
