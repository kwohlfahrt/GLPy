#version 330

out vec4 out_color;

in block {
	flat vec3 color;
} In;

void main(){
	out_color = vec4(In.color, 1) * 0.5 + 0.2;
}
