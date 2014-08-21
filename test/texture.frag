#version 330

out vec4 out_color;
uniform sampler2D tex;

in block {
	vec2 uv;
} In;

void main(){
	out_color = texture(tex, In.uv);
}
