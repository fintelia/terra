#line 2

uniform mat4 modelViewProjection;
uniform vec3 sunDirection;

in vec3 mPosition;
in vec2 mTexcoord;
in vec3 mNormal;

in vec3 vPosition;
in vec3 vColor;
in float vRotation;
in float vScale;
in vec3 vNormal;

out vec3 fColor;
out vec3 fPosition;
out vec2 fTexcoord;

void main() {
	float sr = sin(vRotation);
	float cr = cos(vRotation);
	mat3 rotation = mat3(cr, 0, sr, 0, 1, 0, -sr, 0, cr);

	fPosition = vPosition + (rotation * mPosition) * vScale;
	fColor = vColor*max(dot(vNormal, sunDirection), 0.0);
	fTexcoord = mTexcoord;

	gl_Position = modelViewProjection * vec4(fPosition, 1.0);
}
