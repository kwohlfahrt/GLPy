Transform Feedback
==================

This example shows how to use (OpenGL3-style) transform feedback. This is a good place to start if
you have trouble debugging your shader, as it shows the output of the vertex or geometry shader
stages.

First, the shader. Only a vertex shader is required for this.

.. testcode:: feedback

   vertex_shader = """
   #version 330

   in float number;

   out float xfb;

   void main() {
        xfb = number * 1.2;
   }
   """

   shaders = {'vertex': vertex_shader}

Then, the variables in the shaders are described (a shader parser would be useful!).

.. testcode:: feedback

   from GLPy.GLSL import Variable, UniformBlock, VertexAttribute, FeedbackVarying

   number = VertexAttribute('number', 'float')
   feedback = FeedbackVarying('xfb', 'float')

Some data is set for our geometry. This describes a cube missing its side faces.

.. testcode:: feedback

   from numpy import arange, dtype

   input = arange(10, dtype='float32')

All further steps require an OpenGL context, so one must be created. In this example, we will use
``GLUT`` to create one.

.. testcode:: feedback

   from OpenGL import GLUT, GL

   GLUT.glutInit()
   GLUT.glutInitContextVersion(3, 3)
   GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA | GLUT.GLUT_DEPTH)
   GLUT.glutInitContextProfile(GLUT.GLUT_CORE_PROFILE)
   GLUT.glutInitContextFlags(GLUT.GLUT_FORWARD_COMPATIBLE)
   window_size = (400, 400)
   GLUT.glutInitWindowSize(*window_size) #* (reST syntax highlighting)
   GLUT.glutCreateWindow("GLPy")

We create our program, and VAO for "vertex" specification.

.. testcode:: feedback

   from GLPy import Program, VAO, Buffer

   program = Program.fromSources(shaders, vertex_attributes=[number], xfb_varyings=[feedback])
   vao = VAO(program.vertex_attributes['number'])
   vertex_buffer = Buffer()
   feedback_buffer = Buffer()

Data for the "vertices" are uploaded, and a buffer of the correct size is allocated for the
feedback.

.. testcode:: feedback

   with vertex_buffer.bind(GL.GL_ARRAY_BUFFER):
      vertex_buffer[...] = input
   vao['number'].data = vertex_buffer.items
   with feedback_buffer.bind(GL.GL_TRANSFORM_FEEDBACK_BUFFER):
      feedback_buffer[...] = dtype((feedback.dtype, len(input)))

We can check our buffer contents and see that they match what we set earlier:

.. testcode:: feedback

   with vertex_buffer.bind(GL.GL_ARRAY_BUFFER):
      print((vertex_buffer.data == input).all())

.. testoutput:: feedback

   True

Finally, the following code will output the vertices to our feedback buffer. Note the order of the
bindings.

.. testcode:: feedback

   GL.glDisable(GL.GL_CULL_FACE)

   with vao, program, feedback_buffer.bind(GL.GL_TRANSFORM_FEEDBACK_BUFFER, 0):
      with program.feedback(GL.GL_POINTS):
         GL.glDrawArrays(GL.GL_POINTS, 0, 10)

   GL.glFlush()
   with feedback_buffer.bind(GL.GL_TRANSFORM_FEEDBACK_BUFFER):
      print(feedback_buffer.data)

.. testoutput:: feedback

   [  0.           1.20000005   2.4000001    3.60000014   4.80000019   6.
      7.20000029   8.40000057   9.60000038  10.80000019]
