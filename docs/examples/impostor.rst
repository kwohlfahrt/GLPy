Impostors
=========

This example describes a way to render fake spheres at multiple points, using instances.

The vertex shader places a square at the coordinates specified (pointing towards the origin), and
the fragment shader makes it appear like a perfect sphere.

.. testcode:: sample

	vertex_shader = """
	#version 330

	in vec3 instance_position;
	in vec2 vertex_position;

	out vec2 mapping;
	out vec3 color;

	layout(std140, row_major) uniform Projection {
		mat4 model_camera;
		mat4 camera_clip;
	};

	uniform float size = 0.1;

	void main() {
		vec3 right = vec3(model_camera[0].x, model_camera[1].x, model_camera[2].x);
		vec3 up = vec3(model_camera[0].y, model_camera[1].y, model_camera[2].y);
		vec3 offset = (right * vertex_position.x + up * vertex_position.y) * size;

		mapping = vertex_position;
		color = instance_position;
		gl_Position = camera_clip * model_camera * vec4(instance_position + offset, 1.0);
	}
	"""

	fragment_shader = """
	#version 330

	in vec2 mapping;
	in vec3 color;

	out vec4 frag_color;

	void main() {
		float lensqr = dot(mapping, mapping);
		if (lensqr > 1)
			discard;
		frag_color = vec4(color, 1.0);
	}
	"""

	shaders = {'vertex': vertex_shader, 'fragment': fragment_shader}

Then, the variables in the shaders are described (a shader parser would be useful!).

.. testcode:: sample

	from GLPy.GLSL import Variable, UniformBlock, VertexAttribute

	projection = UniformBlock('Projection', Variable('model_camera', 'mat4'),
	                          Variable('camera_clip', 'mat4'), layout='std140',
	                          matrix_layout='row_major')
	vertex_attribs = [VertexAttribute('instance_position', 'vec3'),
	                  VertexAttribute('vertex_position', 'vec2')]

Some data is set for our geometry. We describe a square to be rendered at each
vertex, and a set of positions where we will draw the squares.

.. testcode:: sample

	from numpy import array, random

	instance_count = 30
	square = array([[-1,-1], [1,-1], [-1, 1], [1, 1]], dtype='int8')
	positions = random.uniform(-2, 2, size=(instance_count, 3)).astype('float32')

All further steps require an OpenGL context, so one must be created. In this example, we will use
``GLUT`` to create one.

.. TODO: Do something about * and syntax highlighting

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

We set up our program constructs.

.. testcode:: sample

	from GLPy import Program, VAO, Buffer

	program = Program.fromSources(shaders, uniform_blocks=[projection],
	                              vertex_attributes=vertex_attribs)
	vao = VAO(*program.vertex_attributes.values()) #*
	projection_buffer = Buffer()
	vertex_buffer = Buffer()
	instance_buffer = Buffer()

And allocate buffers and fill them with data.

.. testcode:: sample

	with projection_buffer.bind(GL.GL_UNIFORM_BUFFER):
		projection_buffer[...] = projection.dtype
	with vertex_buffer.bind(GL.GL_ARRAY_BUFFER):
		vertex_buffer[...] = square
	with instance_buffer.bind(GL.GL_ARRAY_BUFFER):
		instance_buffer[...] = positions

Then the uniforms buffer contents are set, and vertex data is added to the the VAO. Note the setting
of the *vertex attribute divisor* on the instance position. This tells the GL that the attribute
should be advanced once for every *instance* rendered, not for every vertex. A divisor of zero (the
default) advances the attribute once for every vertex.

.. testcode:: sample

	from util import xform
	from math import radians

	with projection_buffer.bind(GL.GL_UNIFORM_BUFFER):
		projection_buffer['model_camera'].data = xform.lookAt((0, 0, 1)).astype('float32')
		projection_buffer['camera_clip'].data = xform.perspective(radians(90)).astype('float32')

	vao['vertex_position'].data = vertex_buffer.items
	vao['instance_position'].data = instance_buffer.items
	vao['instance_position'].divisor = 1

Finally, the following code will display the geometry, with extra code for rotation. Note the use of
``glDrawArraysInstanced`` instead of plain ``glDrawArrays``.

.. testcode:: sample

	program.uniform_blocks['Projection'].binding = 0

	def display():
		 import ctypes as c
		 GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
		 with vao, program:
			 GL.glDrawArraysInstanced(GL.GL_TRIANGLE_STRIP, 0, 4, instance_count)
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
		projection_buffer['model_camera'].data = xform.lookAt((0, 0, 1)).dot(rotation).astype('float32')
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
	with projection_buffer.bind(GL.GL_UNIFORM_BUFFER, program.uniform_blocks['Projection'].binding):
		GLUT.glutMainLoop()
