#line 2

uniform vec3 cameraPosition;
uniform vec3 sunDirection;

in vec3 fPosition;
in vec3 fColor;
in float fRotation;

out vec4 OutColor;

void main() {
	float r = length(fPosition - cameraPosition)/2048;
	float h = random(gl_FragCoord.xy + cameraPosition.xz);

	if(smoothstep(0.8, 0.95, r) > h)
		discard;

	OutColor = vec4(0,0,0, 1);

	float light = fRotation; // Should really be renamed...

	OutColor.rgb = max(vec3(13, 31, 0)/255 + (fColor-1) * 0.1, 0) * light;
	OutColor.rgb = precomputed_aerial_perspective(OutColor.rgb, fPosition, cameraPosition, sunDirection);
}
