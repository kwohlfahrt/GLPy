Tubes
=====

This document renders a series of lines as tubes, with beveled corners. It demonstrates geometry
shaders, transform feedback and basic lighting.

The first program is for constructing our tubes from the input vertices. This requires adjacency
information, so geometry shaders are useful. Other methods are possible and may be faster. A point
normal is required to ensure consistent alignment of tubes along the length of an input line.

.. testcode:: sample

	vertex_shader = """
	#version 330

	in vec3 position;
	in vec3 normal;

	out VertexData {
		vec3 pos;
		vec3 normal;
	} Out;

	void main() {
		Out.pos = position;
		Out.normal = normalize(normal);
	}
	"""

	geometry_shader = """
	#version 330
	#extension GL_ARB_gpu_shader5 : require

	layout(lines_adjacency) in;
	layout(triangle_strip, max_vertices=128) out;

	layout(std140) uniform Tube {
		int npts;
		float radius;
	};
	layout(std140, row_major) uniform Projection {
		mat4 model_camera;
		mat4 camera_clip;
	};

	in VertexData {
		vec3 pos;
		vec3 normal;
	} In[];

	out VertexData {
		vec3 pos;
		vec3 normal;
	} Out;

	out vec3 normal;

	#define M_PI 3.1415926535897932384626433832795
	// Rodrigues' rotation formula
	vec3 rotRodrigues(vec3 vector, vec3 axis, float angle){
		return ( vector * cos(angle) + cross(axis, vector) * sin(angle)
		       + axis * dot(axis, vector) * (1 - cos(angle)) );
	}

	void main() {
		vec3 segments[3];
		float segment_lengths[segments.length()];
		for (int i = 0; i < segments.length(); ++i){
			vec3 segment = In[i].pos - In[i + 1].pos;
			segment_lengths[i] = length(segment);
			segments[i] = normalize(segment);
		}

		// Normal to the plane that bisects the angle between two segments
		vec3 bisection_normals[2];
		for (int i = 0; i < bisection_normals.length(); ++i){
			bisection_normals[i] = cross(segments[i + 1] - segments[i],
			                             cross(segments[i + 1], segments[i]));
		}

		for (int i = 0; i <= npts; ++i){
			vec3 norm = rotRodrigues(In[1].normal, segments[1], 2 * M_PI * i / npts);
			vec3 offset = norm * radius;
			for (int end = 0; end < 2; ++end){
				vec3 tube_axis = -segments[1] * (end * 2 - 1);
				float segment_length = min(segment_lengths[1], segment_lengths[end * 2]);

				float overlap = ( dot(bisection_normals[end], offset)
				                / dot(bisection_normals[end], tube_axis));
				Out.pos = In[end + 1].pos + offset - clamp(overlap, 0, segment_length) * tube_axis;

				Out.normal = norm;
				gl_Position = camera_clip * model_camera * vec4(Out.pos, 1.0);
				EmitVertex();
			}
		}
		EndPrimitive();
	}
	"""

	fragment_shader = """
	#version 330

	out vec4 frag_color;

	in VertexData {
		vec3 pos;
		vec3 normal;
	} In;

	void main() {
		float intensity = max(dot(In.normal, vec3(0.0, 0.0, 1.0)), 0) + 0.05;
		frag_color = vec4(intensity * vec3(1.0, 1.0, 1.0), 1.0);
	}
	"""

	shaders = {'vertex': vertex_shader, 'geometry': geometry_shader, 'fragment': fragment_shader}

Then, the variables in the shaders are described (a shader parser would be useful!).

.. testcode:: sample

	from GLPy.GLSL import Variable, UniformBlock, VertexAttribute

	projection = UniformBlock('Projection',
	                          Variable('model_camera', 'mat4'), Variable('camera_clip', 'mat4'),
	                          layout='std140', matrix_layout='row_major')
	tube_params = UniformBlock('Tube',
	                           Variable('npts', 'int'), Variable('radius', 'float'),
	                           layout='std140')
	vertex_attribs = [VertexAttribute('position', 'vec3'), VertexAttribute('normal', 'vec3')]

Some data is set for our geometry. This describes random points as vertices, and then calculates
normals for each point.

.. testcode:: sample

	from math import cos, acos, sin
	from numpy import array, empty, cross, dot
	from numpy.linalg import norm
	from numpy import random

	npoints = 10
	positions = random.uniform(-1, 1, size=(npoints, 3)).astype('float32')
	normals = empty(positions.shape, dtype=positions.dtype)

	lines = positions[:-1] - positions[1:]
	normals[0] = cross(lines[0], [0, 1, 0])
	normals[0] /= norm(normals[0])
	for i, ls in enumerate(zip(lines[:-1], lines[1:]), start=1):
		theta = acos(dot(ls[0], ls[1]) / (norm(ls[0]) * norm(ls[1])))
		k = cross(*ls) #*
		k /= norm(k)
		normals[i] = ( normals[i-1] * cos(theta) + cross(k, normals[i-1]) * sin(theta)
		             + k * dot(k, normals[i-1]) * (1 - cos(theta)))
	normals = array(normals, dtype=('float32')).astype('float32')

All further steps require an OpenGL context, so one must be created. In this example, we will use
``GLUT`` to create one.

.. testcode:: sample

	from OpenGL import GLUT, GL

	GLUT.glutInit()
	GLUT.glutInitContextVersion(3, 3)
	GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA | GLUT.GLUT_DEPTH)
	GLUT.glutInitContextProfile(GLUT.GLUT_CORE_PROFILE)
	GLUT.glutInitContextFlags(GLUT.GLUT_FORWARD_COMPATIBLE)
	window_size = (400, 400)
	GLUT.glutInitWindowSize(*window_size) #* (reST syntax highlighting)
	GLUT.glutCreateWindow("GLPy")

The various OpenGL constructs, such as vertex arrays and buffers have their own classes.

.. testcode:: sample

	from GLPy import Program, VAO, Buffer

	program = Program.fromSources(shaders, uniform_blocks=[projection, tube_params],
	                              vertex_attributes=vertex_attribs)
	for i, block in enumerate(program.uniform_blocks.values()):
		block.binding = i

	vao = VAO(*program.vertex_attributes.values()) #*
	projection_buffer = Buffer()
	tube_buffer = Buffer()
	vertex_buffer = Buffer()
	normal_buffer = Buffer()

An empty buffer is allocated for the projection uniforms, and data is uploaded directly for the
vertex buffer.

.. testcode:: sample

	with projection_buffer.bind(GL.GL_UNIFORM_BUFFER):
		projection_buffer[...] = projection.dtype
	with tube_buffer.bind(GL.GL_UNIFORM_BUFFER):
		tube_buffer[...] = tube_params.dtype
	with vertex_buffer.bind(GL.GL_ARRAY_BUFFER):
		vertex_buffer[...] = positions
	with normal_buffer.bind(GL.GL_ARRAY_BUFFER):
		normal_buffer[...] = normals

Then the uniforms buffer contents are set, and vertex data is added to the the VAO.

.. testcode:: sample

	from util import xform
	from math import radians

	camera_start = (0, 0, 3)
	with projection_buffer.bind(GL.GL_UNIFORM_BUFFER):
		projection_buffer['model_camera'].data = xform.lookAt(camera_start).astype('float32')
		projection_buffer['camera_clip'].data = xform.perspective(radians(90)).astype('float32')
	with tube_buffer.bind(GL.GL_UNIFORM_BUFFER):
		tube_buffer['radius'].data = array([0.1], dtype='float32')
		tube_buffer['npts'].data = array([10], dtype='int32')

	vao['position'].data = vertex_buffer.items
	vao['normal'].data = normal_buffer.items

Finally, the following code will display the geometry

.. testcode:: sample

	def display():
		import ctypes as c
		GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
		with vao, program:
			GL.glDrawArrays(GL.GL_LINE_STRIP_ADJACENCY, 0, npoints)
		GLUT.glutSwapBuffers()
	GLUT.glutDisplayFunc(display)

	GL.glEnable(GL.GL_DEPTH_TEST)
	GL.glDisable(GL.GL_CULL_FACE)
	GL.glClearColor(0, 0, 0, 1)

	from util.arcball import ArcBall

	centre = tuple(w/2 for w in window_size)
	axes = (centre[0], -centre[1])
	arcball = ArcBall(centre, axes)

	def updateRotation(rotation):
		projection_buffer['model_camera'].data = ( xform.lookAt(camera_start).dot(rotation)
		                                          .astype('float32') )
		display()

	def mousebutton(button, state, x, y):
		global arcball
		if state == GLUT.GLUT_DOWN:
			arcball.startRotation(x, y)
			updateRotation(arcball.totalRotation())
		elif state == GLUT.GLUT_UP:
			arcball.finishRotation()

	def mousemove(x, y):
		global arcball
		arcball.updateRotation(x, y)
		updateRotation(arcball.totalRotation())

	def keypress(key, x, y):
		if key == b'q':
			GLUT.glutLeaveMainLoop()
		display()

	GLUT.glutKeyboardFunc(keypress)
	GLUT.glutMouseFunc(mousebutton)
	GLUT.glutMotionFunc(mousemove)

::
	with tube_buffer.bind(GL.GL_UNIFORM_BUFFER, program.uniform_blocks['Tube'].binding), projection_buffer.bind(GL.GL_UNIFORM_BUFFER, program.uniform_blocks['Projection'].binding):
		GLUT.glutMainLoop()
