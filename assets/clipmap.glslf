#version 450 core

in vec2 v_TexCoord;
out vec4 OutColor;

uniform sampler2D t_color;

void main() {
    OutColor = vec4(1,0,0,1);
}
