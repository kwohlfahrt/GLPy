Sample
======

This document describes a 'Hello cube!' program.

First, the shaders are written. This shader defines a uniform variable, a
uniform block and one vertex attribute.

.. testcode:: sample

   vertex_shader = """
   #version 330

   in vec3 position;

   layout(std140, row_major) uniform Projection {
        mat4 model_camera;
        mat4 camera_clip;
   };

   out block {
        vec3 color;
   } Out;

   void main() {
        Out.color = (position + 2) / 4;
        gl_Position = camera_clip * model_camera * vec4(position, 1.0);
   }
   """

   fragment_shader = """
   #version 330

   out vec4 frag_color;

   in block {
        vec3 color;
   } In;

   void main() {
        frag_color = vec4(In.color, 1);
   }
   """

   shaders = {'vertex': vertex_shader, 'fragment': fragment_shader}

Then, the variables in the shaders are described (a shader parser would be useful!).

.. testcode:: sample

   from GLPy.GLSL import Variable, UniformBlock, VertexAttribute

   projection = UniformBlock('Projection', Variable('model_camera', 'mat4'),
                             Variable('camera_clip', 'mat4'), layout='std140',
                             matrix_layout='row_major')
   position = VertexAttribute('position', 'vec3')

Some data is set for our geometry. This describes a cube missing its side faces.

.. testcode:: sample

   from numpy import array

   cube = array([[-1,-1,-1], [1,-1,-1], [-1, 1,-1], [1, 1,-1],
                 [-1,-1, 1], [1,-1, 1], [-1, 1, 1], [1, 1, 1]],
                 dtype='int8')
   indices = array([0, 1, 2, 3, 6, 7, 4, 5, 0, 1], dtype='uint8')

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

The various OpenGL constructs, such as vertex arrays and buffers have their own classes.

.. testcode:: sample

   from GLPy import Program, VAO, Buffer

   program = Program.fromSources(shaders, uniform_blocks=[projection], vertex_attributes=[position])
   vao = VAO(program.vertex_attributes['position'])
   projection_buffer = Buffer()
   element_buffer = Buffer()
   vertex_buffer = Buffer()

An empty buffer is allocated for the projection uniforms, and data is uploaded directly for the
element and vertex buffers.

.. testcode:: sample

   with projection_buffer.bind(GL.GL_UNIFORM_BUFFER):
      projection_buffer[...] = projection.dtype
   with element_buffer.bind(GL.GL_ELEMENT_ARRAY_BUFFER):
      element_buffer[...] = indices
   with vertex_buffer.bind(GL.GL_ARRAY_BUFFER):
      vertex_buffer[...] = cube

Then the uniforms buffer contents are set, and vertex data is added to the the VAO.

.. testcode:: sample

   from util import xform
   from math import radians

   with projection_buffer.bind(GL.GL_UNIFORM_BUFFER):
      projection_buffer['model_camera'].data = xform.lookAt((0, 0, 3)).astype('float32')
      projection_buffer['camera_clip'].data = xform.perspective(radians(90)).astype('float32')

   vao.element_buffer = element_buffer
   vao['position'].data = vertex_buffer.items

Finally, the following code will display the geometry

.. testcode:: sample

   program.uniform_blocks['Projection'].binding = 0

   def display():
       import ctypes as c
       GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
       with vao, program:
          GL.glDrawElements(GL.GL_TRIANGLE_STRIP, len(indices), GL.GL_UNSIGNED_BYTE, c.c_void_p(0))
       GLUT.glutSwapBuffers()
   GLUT.glutDisplayFunc(display)

   GL.glEnable(GL.GL_DEPTH_TEST)
   GL.glDisable(GL.GL_CULL_FACE)
   GL.glClearColor(0, 0, 0, 1)

With a little additional effort, it can also be made interactive. In the following example, the
mouse will rotate the cube

.. testcode:: sample

   from util.arcball import ArcBall

   arcball = ArcBall(window_size, (window_size[0], -window_size[1]))

   def updateRotation(rotation):
      projection_buffer['model_camera'].data = xform.lookAt((0, 0, 3)).dot(rotation).astype('float32')
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

   with projection_buffer.bind(GL.GL_UNIFORM_BUFFER, program.uniform_blocks['Projection'].binding):
      GLUT.glutMainLoop()
