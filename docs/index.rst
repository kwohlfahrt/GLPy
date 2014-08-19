GLPy
++++

GLPy is a python module based on `PyOpenGL`_ that gives a nice, class-based
wrapper around common OpenGL functions. It aims to allow access (within reason)
to most native OpenGL functionality.

Installation
============

Installation via ``pip`` will soon be supported. Not yet though.

Requirements
------------

`PyOpenGL`_ is required.

Sample
======

A simple sample program looks like this:

.. testsetup:: sample

   import numpy
   from OpenGL import GL, GLUT
   from math import radians

   from GLPy import GLSLVar, Program, VAO, UniformBlock, UniformBuffer, ElementBuffer, VertexBuffer

   from util import xform
   from util.arcball import ArcBall

First, the shaders are written. This shader defines a uniform variable, a
uniform block and one vertex attribute.

.. testcode:: sample

   vertex_shader = """
   #version 330

   in vec4 position;

   layout(shared, row_major) uniform Projection {
   	uniform mat4 model_camera_xform;
   	uniform mat4 camera_clip_xform;
   };

   uniform bool red = false;

   out block {
   	flat vec3 color;
   } Out;

   void main(){
   	if (red){
   		Out.color = vec3(1, 0, 0);
   	} else {
   		Out.color = (position.xyz + 2) / 4;
   	}
   	gl_Position = camera_clip_xform
   	            * model_camera_xform
   	            * position;
   }
   """

   fragment_shader = """
   #version 330

   out vec4 out_color;

   in block {
   	flat vec3 color;
   } In;

   void main(){
   	out_color = vec4(In.color, 1);
   }
   """

   shaders = {'vertex': vertex_shader, 'fragment': fragment_shader}

Then, the variables in the shaders are described (a shader parser would be
useful!).

.. testcode:: sample

   projection_uniforms = [ GLSLVar('model_camera_xform', 'mat4')
                         , GLSLVar('camera_clip_xform', 'mat4') ]
   uniforms = [GLSLVar('red', 'bool')]
   attributes = [GLSLVar('position', 'vec4')]

Some data is set for our geometry. This describes a cube missing its side
faces.

.. testcode:: sample

   cube = numpy.array([ [-1,-1,-1], [1,-1,-1], [-1, 1,-1], [1, 1,-1]
                      , [-1,-1, 1], [1,-1, 1], [-1, 1, 1], [1, 1, 1]]
                      , dtype='float32')
   indices = numpy.array([0, 1, 2, 3, 6, 7, 4, 5, 0, 1], dtype='uint32')

All further steps require an OpenGL context, so one must be created. In this
example, we will use ``GLUT`` to create one.

.. testcode:: sample

   GLUT.glutInit()
   GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA)
   window_size = (400, 400)
   GLUT.glutInitWindowSize(window_size[0], window_size[1])
   GLUT.glutCreateWindow("GLPy")

   GL.glClearColor(0, 0, 0, 1)
   GL.glEnable(GL.GL_DEPTH_TEST)
   GL.glDisable(GL.GL_CULL_FACE)

The various OpenGL constructs, such as vertex arrays and buffers have their own
constructors.

.. testcode:: sample

   vao = VAO(attributes[0])
   program = Program(shaders, uniforms=uniforms, attributes=vao.attributes)
   projection_block = UniformBlock( 1, program, "Projection"
                                  , projection_uniforms[0], projection_uniforms[1])
   projection_buffer = UniformBuffer(projection_block)

   element_buffer = ElementBuffer()
   element_buffer[:] = indices

   vertex_buffer = VertexBuffer(vao.attributes[0])

Then the uniforms and buffer contents are set, and vertex data is added to the the VAO.

.. testcode:: sample
   
   projection_buffer.blocks[0].members[0].data = xform.lookAt((0, 0, 3)) # model_camera_xform
   projection_buffer.blocks[0].members[1].data = xform.perspective(radians(90)) # camera_clip_xform
   program.uniforms['red'].data = False

   # We only want to set 3 of the vec4 components
   vertex_buffer.blocks[0].tracks[0].components = 3
   vertex_buffer[:] = cube,
   vao.attributes[0].data = vertex_buffer.blocks[0].tracks[0]

   vao.elements = element_buffer

Finally, the following code will display the geometry::

   def display():
       GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
       with vao, program:
          GL.glDrawElements(GL.GL_TRIANGLE_STRIP, len(indices), GL.GL_UNSIGNED_INT, None)
       GLUT.glutSwapBuffers()

   GLUT.glutDisplayFunc(display)
   GLUT.glutMainLoop()

With a little additional effort, it can also be made interactive. In the
following example, the mouse will rotate the cube and pressing 'r' will toggle
between different color schemes.

::

   arcball = ArcBall(window_size, (window_size[0], -window_size[1]))

   def updateRotation(rotation):
      projection_buffer.blocks[0][0].data = xform.lookAt((0, 0, 3)).dot(rotation)
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
      if key == b'r':
         program.uniforms['red'].data = not program.uniforms['red'].data
      display()

   GLUT.glutKeyboardFunc(keypress)
   GLUT.glutMouseFunc(mousebutton)
   GLUT.glutMotionFunc(mousemove)

.. toctree::

   api
