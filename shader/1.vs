#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNom;

out VS_OUT {
    vec3 FragPos;
    vec4 FragPosLightSpace;
} vs_out;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 lightSpaceMatrix;

out vec3 normal;

void main()
{
    vs_out.FragPos = aPos;
    vs_out.FragPosLightSpace = lightSpaceMatrix * vec4(vs_out.FragPos, 1.0);
    normal = aNom;
    gl_Position =projection * view * vec4(aPos,1.0f);
}