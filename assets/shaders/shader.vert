#version 450
#extension GL_ARB_separate_shader_objects : enable

const vec2 POSITIONS[3] = vec2[](
    vec2(-1.0, 1.0),
    vec2(1.0, 1.0),
    vec2(0.0, -1.0)
);

const vec3 COLORS[3] = vec3[](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);

layout(location = 0) out vec3 oColor;

void main() {
    vec2 position = POSITIONS[gl_VertexIndex];
    oColor = COLORS[gl_VertexIndex];

    gl_Position = vec4(position.x, position.y, 0.0, 1.0);
}
