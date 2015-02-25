#version 330
#extension GL_ARB_shading_language_420pack : enable
//#extension GL_ARB_explicit_uniform_location : enable

struct TheStruct {
	vec3 first;
	vec4 second;
	mat4 third;
};
 
uniform int aScalarUniform = 2;
uniform vec3 aUniform = vec3(1.0, 2.0, 3.0);
uniform TheStruct aStructUniform;
uniform mat4 matrixArrayUniform[2] = {mat4(1), mat4(2)};
//layout(location=1) uniform TheStruct uniformArrayOfStructs[2];
uniform TheStruct uniformArrayOfStructs[2];

void main(){
	vec3 pos = aUniform * aScalarUniform
	         + aStructUniform.first
	         + aStructUniform.second.xyz;

	mat4 total_xform = mat4(1);
	for (int i = 0; i < matrixArrayUniform.length(); i++){
		total_xform *= matrixArrayUniform[i];
	}

	for (int i = 0; i < uniformArrayOfStructs.length(); i++){
		TheStruct s = uniformArrayOfStructs[i];
		pos += s.first;
		pos += s.second.xyz;
		total_xform *= s.third;
	}
	
	gl_Position = total_xform * vec4(pos, 1.0);
}
