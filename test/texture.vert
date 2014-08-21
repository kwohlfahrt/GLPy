#version 330

in vec4 position;

out block {
	vec2 uv;
} Out;

void main(){
	Out.uv = position.xy;
	gl_Position =  position;
}
