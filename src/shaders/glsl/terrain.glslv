#line 2
uniform int resolution;
uniform mat4 modelViewProjection;
uniform vec3 cameraPosition;

uniform sampler2DArray heights;

in vec2 vPosition;
in float vSideLength;
in float vMinDistance;
in vec3 heightsOrigin;
in vec2 textureOrigin;
in float textureStep;
in vec2 parentTextureOrigin;
in float parentTextureStep;
in vec2 colorsLayer;
in vec2 normalsLayer;
in vec2 splatsLayer;

out vec3 fPosition;
out vec2 fTexcoord;
out vec2 fParentTexcoord;
out vec2 fColorsLayer;
out vec2 fNormalsLayer;
out vec2 fSplatsLayer;
out float fMorph;

void main() {
	vec3 position = vec3(0);
	ivec2 iPosition = ivec2((gl_VertexID) % (resolution+1),
							(gl_VertexID) / (resolution+1));

	vec3 cp = cameraPosition;
	cp.y = 0;

	position.xz = vec2(iPosition)
	    * (vSideLength / (resolution)) + vPosition;
	float morph = 1 - smoothstep(0.7, 0.95, distance(position, cameraPosition) / vMinDistance);
	morph = min(morph * 2, 1);
	if(colorsLayer.y < 0)
		morph = 1;

	position.y = texture(heights,
						 heightsOrigin + vec3(vec2(iPosition + 0.5) / textureSize(heights, 0).xy, 0)).r;

	ivec2 morphTarget = (iPosition / 2) * 2;
	float morphHeight = texture(heights, heightsOrigin + vec3(vec2(morphTarget + 0.5) / textureSize(heights, 0).xy, 0)).r;

	vec2 nPosition = mix(vec2(morphTarget), vec2(iPosition), morph);

	position.y = mix(morphHeight, position.y, morph);
	position.xz = nPosition * (vSideLength / (resolution)) + vPosition;

	fPosition = position;
	fTexcoord = textureOrigin + nPosition * textureStep;
	fParentTexcoord = parentTextureOrigin + nPosition * parentTextureStep;
	fColorsLayer = colorsLayer;
	fNormalsLayer = normalsLayer;
	fSplatsLayer = splatsLayer;
	fMorph = morph;
	gl_Position = modelViewProjection * vec4(position, 1.0);
}
