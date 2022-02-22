#version 450 core
#include "declarations.glsl"

layout(location = 0) out vec2 texcoord;
layout(location = 1) out float magnitude;
layout(location = 2) out vec4 position;

layout(set = 0, binding = 0, std140) uniform UniformBlock {
    Globals globals;
};

struct Star {
    float declination;
	float ascension;
	float magnitude;
	float padding;
};
layout(binding = 1) readonly buffer Stars {
	Star starfield[];
};

void main() {
	Star star = starfield[gl_VertexIndex / 6];

	if(gl_VertexIndex % 6 == 0) texcoord = vec2(0, 0);
	if(gl_VertexIndex % 6 == 1) texcoord = vec2(1, 0);
	if(gl_VertexIndex % 6 == 2) texcoord = vec2(0, 1);
	if(gl_VertexIndex % 6 == 3) texcoord = vec2(1, 1);
	if(gl_VertexIndex % 6 == 4) texcoord = vec2(0, 1);
	if(gl_VertexIndex % 6 == 5) texcoord = vec2(1, 0);

	vec4 direction = vec4(
		cos(star.ascension + globals.sidereal_time) * cos(star.declination),
		sin(star.ascension + globals.sidereal_time) * cos(star.declination),
		sin(star.declination),
		1e-15);

	magnitude = star.magnitude;

	gl_Position = globals.view_proj * direction;
	gl_Position.xy += (texcoord-0.5) * gl_Position.w * 4.0/vec2(globals.screen_width, globals.screen_height);
	position = gl_Position;
}
