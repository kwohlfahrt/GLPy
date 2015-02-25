#version 330

in vec4 position;
in vec3 color;
in mat2x4 foo;
in mat3 bar[2];
in vec2 baz;

out block {
	flat vec3 color;
} Out;

void main(){
	Out.color = bar[1] * bar[0] * color;
	gl_Position =  ( foo * baz ) + position;
}
