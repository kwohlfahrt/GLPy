from OpenGL import GL

from .vertex import ProgramVertexAttribute
from .uniform_block import ProgramUniformBlock

from contextlib import contextmanager

import ctypes as c

shader_types = { 'vertex': GL.GL_VERTEX_SHADER
			   , 'fragment': GL.GL_FRAGMENT_SHADER
			   , 'geometry': GL.GL_GEOMETRY_SHADER
			   , 'tesselation control': GL.GL_TESS_CONTROL_SHADER
			   , 'tesselation evaluation': GL.GL_TESS_EVALUATION_SHADER }
shader_types.update({t: t for t in shader_types.values()})

class Shader:
	def __init__(self, source, shader_type):
		self.shader_type = shader_types[shader_type]
		self.handle = GL.glCreateShader(self.shader_type)
		if self.handle == 0:
			raise RuntimeError("Failed to create shader.")

		GL.glShaderSource(self.handle, source)
		GL.glCompileShader(self.handle)
		if GL.glGetShaderiv(self.handle, GL.GL_COMPILE_STATUS) == GL.GL_FALSE:
			log = GL.glGetShaderInfoLog(self.handle).decode()
			raise RuntimeError("Failed to compile {}: \n\n{}".format(self.shader_type, log))

	def delete(self):
		'''Delete the shader to free up GL resources.

		.. warning::

		   Do not use the shader object after running this method.
		'''

		GL.glDeleteShader(self.handle)

class Program:
	"""An OpenGL program.
	
	:param shaders: The shaders. It is good practice to delete them after creating the program.
	:type shaders: [:py:class:`Shader`]
	:param attributes: The vertex attributes used in the program
	:type attributes: [:py:class:`.VertexAttribute`]
	:param uniform_blocks: The uniform blocks defined in the program
	:type uniform_blocks: [:py:class:`.UniformBlock`]
	"""

	def __init__(self, shaders, vertex_attributes=None, uniform_blocks=None,
	             xfb_varyings=None, xfb_mode=GL.GL_INTERLEAVED_ATTRIBS):
		self.handle = GL.glCreateProgram()
		if self.handle == 0:
			raise RuntimeError("Failed to create OpenGL program.")

		for shader in shaders:
			GL.glAttachShader(self.handle, shader.handle)

		self.xfb_varyings = xfb_varyings
		if xfb_varyings is not None:
			varyings = (c.c_char_p * len(xfb_varyings))(*(v.name.encode() for v in self.xfb_varyings))
			varyings = c.cast(varyings, c.POINTER(c.POINTER(c.c_char)))
			GL.glTransformFeedbackVaryings(self.handle, len(xfb_varyings), varyings, xfb_mode)
		self._xfb_mode = xfb_mode

		GL.glLinkProgram(self.handle)
		if GL.glGetProgramiv(self.handle, GL.GL_LINK_STATUS) == GL.GL_FALSE:
			log = GL.glGetProgramInfoLog(self.handle).decode()
			GL.glDeleteProgram(self.handle)
			raise RuntimeError("Failed to link program: \n\n{}".format(log))

		self.uniform_blocks = { ub.name: ProgramUniformBlock.fromUniformBlock(self, ub)
		                        for ub in uniform_blocks or []}
		self.vertex_attributes = { v.name: ProgramVertexAttribute.fromVertexAttribute(self, v)
		                           for v in vertex_attributes or [] }

	@property
	def xfb_mode(self):
		return self._xfb_mode

	@xfb_mode.setter
	def xfb_mode(self, xfb_mode):
		varyings = (c.c_char_p * len(xfb_varyings))(*(v.name for v in self.xfb_varyings))
		varyings = c.cast(varyings, c.POINTER(c.POINTER(c.c_char)))
		GL.glTransformFeedbackVaryings(self.handle, len(xfb_varyings), varyings, xfb_mode)
		self._xfb_mode = xfb_mode

	@classmethod
	def fromSources(cls, sources, **kwargs):
		shaders = []
		try:
			for shader_type, shader in sources.items():
				shaders.append(Shader(shader, shader_type))
			program = cls(shaders, **kwargs)
		finally:
			for shader in shaders:
				shader.delete()
		return program

	def __enter__(self):
		'''Programs provide a context manager that binds and then unbinds them.

		.. _program-bind-warning:
		.. warning::

		   It is not allowed to bind multiple programs (or one program multiple times).
		   
		   Methods that bind a program will be documented.
		'''
		GL.glUseProgram(self.handle)
	
	def __exit__(self, ty, val, tr):
		GL.glUseProgram(0)

	@contextmanager
	def feedback(self, mode=GL.GL_POINTS):
		GL.glBeginTransformFeedback(mode)
		yield
		GL.glEndTransformFeedback()
