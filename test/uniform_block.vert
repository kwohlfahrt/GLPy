#version 330

uniform Projection {
	mat4 xform;
	bool origin;
};

void main(){
	vec4 pos;
	if (origin){
		pos = vec4(0, 0, 0, 1);
	} else {
		pos = vec4(1, 1, 1, 1);
	}
	gl_Position = xform * pos;
}
