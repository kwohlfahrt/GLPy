#version 450

in vec4 color1;
noperspective in vec4 color2;
out vec4 color;

uniform vec4 multiplier;
uniform bool cond;
uniform int switch_cond;

struct S {
    bool b;
    vec4 v[5];
    int i;
};
uniform S s;

void main()
{
    vec4 scale = vec4(1.0, 1.0, 2.0, 1.0);

    if (cond)
        color = color1 + s.v[2];
    else
        color = sqrt(color2) * scale;

    for (int i = 0; i < 4; ++i)
        color *= multiplier;

    switch(switch_cond){
    case (0):
        color += 3;
    case (1):
        color += 10;
    case (2):
        color += 8;
    case (4):
    default:
        color += 0;
    }
}
