#version 450 core

in ivec3 pos;

out vec3 vPosition;

void main() {
    vPosition = pos;
}
