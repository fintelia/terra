#line 2
uniform mat4 modelViewProjection;

uniform sampler2DArray normals;

in vec3 fPosition;
in vec3 fNormalsTexcoord;

out vec4 OutColor;

void main() {
	vec3 normal = normalize(texture(normals, fNormalsTexcoord).rgb * 2.0 - 1.0);
	float nDotL = dot(normal, normalize(vec3(0,1,1)));
	OutColor = vec4(vec3(nDotL), 1);
}
