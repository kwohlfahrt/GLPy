#version 330

uniform mat4 xform = mat4(1);
uniform bool origin = false;
uniform int is[3] = int[3](1, 2, 3);

void main(){
	vec3 pos = vec3(is[0], is[1], is[2]);
	if (origin){
		pos = vec3(0, 0, 0);
	}
	gl_Position = xform
	            * vec4(pos, 1);
}
