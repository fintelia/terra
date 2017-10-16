#line 2
uniform mat4 modelViewProjection;

uniform sampler2DArray colors;
uniform sampler2DArray normals;

in vec3 fPosition;
in vec2 fTexcoord;
in float fColorsLayer;
in float fNormalsLayer;

out vec4 OutColor;

void main() {
	vec4 color = texture(colors, vec3(fTexcoord, fColorsLayer));
	OutColor = vec4(color.rgb, 1);

	if(fNormalsLayer >= 0) {
		vec3 normal = normalize(texture(normals, vec3(fTexcoord, fNormalsLayer)).xzy * 2.0 - 1.0);
		OutColor.rgb *= dot(normal, normalize(vec3(0,1,1)));
	} else {
		OutColor.rgb *= color.a;
	}
}
