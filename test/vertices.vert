#version 330

in vec4 position;
in vec3 color;

out block {
	flat vec3 color;
} Out;

void main(){
	Out.color = color;
	gl_Position =  position;
}
