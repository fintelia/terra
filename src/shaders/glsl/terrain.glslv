#line 2
uniform int resolution;
uniform mat4 modelViewProjection;
uniform vec3 cameraPosition;

uniform sampler2DArray heights;

in vec2 vPosition;
in float vSideLength;
in float vMinDistance;
in vec3 heightsOrigin;

out vec3 fPosition;

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
	position.y = texture(heights,
						 heightsOrigin + vec3(vec2(iPosition) / (textureSize(heights, 0).xy - vec2(1,1)), 0)).r;

	ivec2 morphTarget = (iPosition / 2) * 2;
	float morphHeight = texture(heights, heightsOrigin + vec3(vec2(morphTarget) / (textureSize(heights, 0).xy - vec2(1,1)), 0)).r;

	position.y = mix(morphHeight, position.y, morph);
	position.xz = mix(vec2(morphTarget), vec2(iPosition), morph)
		* (vSideLength / (resolution)) + vPosition;

	fPosition = position;
	gl_Position = modelViewProjection * vec4(position, 1.0);
}
