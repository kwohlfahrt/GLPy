#version 330

// For explicit block binding
# extension GL_ARB_shading_language_420pack : require

struct Foo {
	float f;
	vec4 v4;
};

layout(std140, binding=2) uniform AnonymousUB {
	mat4 m4;
	bool b;
};

layout(packed) uniform InstancedUB {
	float f;
	vec4 v4;
	Foo foo;
} ubinstance;

layout(packed) uniform OptimizedUB {
	uint ui;
	int i;
	Foo foo[2];
} uboptimized;

// Make sure all uniform members are used
void main(){
	vec4 pos;
	if (b){
		pos = vec4(0, 0, 0, 1);
	} else {
		pos = vec4(1, 1, 1, 1);
	}
	gl_Position = m4 * pos
	            + ubinstance.v4
		      * ubinstance.f * ubinstance.foo.f
	              * uboptimized.ui * uboptimized.foo[0].f;
}
