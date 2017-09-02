#line 2
uniform int resolution;
uniform mat4 modelViewProjection;
uniform vec3 cameraPosition;

in vec2 vPosition;
in float vSideLength;
in float vMinDistance;

const ivec2 OFFSETS[6] = ivec2[6](
	ivec2(0,0),
	ivec2(1,0),
	ivec2(1,1),
	ivec2(0,1),
	ivec2(0,0),
	ivec2(1,1));

void main() {
	vec3 position = vec3(0);
	ivec2 iPosition = ivec2((gl_VertexID/6) % (resolution),
							(gl_VertexID/6) / (resolution))
		+ OFFSETS[gl_VertexID % 6];

	position.xz = vec2(iPosition)
	    * (vSideLength / (resolution)) + vPosition;

	float morph = 1 - smoothstep(0.7, 1.0, distance(position, cameraPosition) / vMinDistance);
	ivec2 morphTarget = (iPosition / 2) * 2;
	position.xz = mix(vec2(morphTarget), vec2(iPosition), morph)
		* (vSideLength / (resolution)) + vPosition;

	gl_Position = modelViewProjection * vec4(position, 1.0);
}
