#version 330

uniform Projection {
	mat4 xform;
	bool origin;
};

void main(){
	vec3 pos = vec3(1, 1, 1);
	if (origin){
		pos = vec3(0, 0, 0);
	}
	gl_Position = xform
	            * vec4(pos, 1);
}
