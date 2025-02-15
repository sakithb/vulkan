#version 450

layout(location = 0) in vec2 inPos;
layout(location = 1) in vec3 inCol;
layout(location = 2) in vec2 inTex;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTex;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPos, 0.0, 1.0);
    fragColor = inCol;
    fragTex = inTex;
}
