from OpenGL import GL

from .uniform import UniformAttribute
from .vertex import VertexAttribute
from . import datatypes

from itertools import product
from collections import namedtuple
from functools import partial

SHADER_TYPES = { 'vertex': GL.GL_VERTEX_SHADER
               , 'fragment': GL.GL_FRAGMENT_SHADER
               , 'geometry': GL.GL_GEOMETRY_SHADER
               , 'tesselation control': GL.GL_TESS_CONTROL_SHADER
               , 'tesselation evaluation': GL.GL_TESS_EVALUATION_SHADER }

class Program:
	"""An OpenGL program."""


	def __init__(self, sources, attributes=None, uniforms=None, shared_uniforms=None):
		attributes = attributes or []
		uniforms = uniforms or []
		shared_uniforms = shared_uniforms or []

		self.bound = 0
		self.handle = GL.glCreateProgram()
		if self.handle == 0:
			raise RuntimeError("Failed to create OpenGL program.")

		for shader_type, src in sources.items():
			gl_shader_type = SHADER_TYPES[shader_type]
			shader = GL.glCreateShader(gl_shader_type)
			GL.glShaderSource(shader, src)
			GL.glCompileShader(shader) 
			if GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS) == GL.GL_FALSE:
				log = GL.glGetShaderInfoLog(shader).decode()
				raise RuntimeError("Failed to compile {} shader: \n\n{}".format(shader_type, log))
			GL.glAttachShader(self.handle, shader)

		# TODO: Enable getting of dynamically bound vertex attributes
		for attribute in attributes:
			if attribute.location is not None:
				GL.glBindAttribLocation(self.handle, attribute.location, attribute.name)

		GL.glLinkProgram(self.handle)
		if GL.glGetProgramiv(self.handle, GL.GL_LINK_STATUS) == GL.GL_FALSE:
			log = GL.glGetProgramInfoLog(self.handle).decode()
			raise RuntimeError("Failed to link program: \n\n{}".format(log))

		self.uniforms = {u.name: UniformAttribute.fromGLSLVar(self, u) for u in uniforms}
	
	def __enter__(self):
		'''Program provide a context manager that keep track of how many times the program has been
		bound and unbound. Grouping operations on a program within a context where it is bound
		reduce unnecessary binding and unbinding.

		.. _program-bind-warning:
		.. warning::
		   It is not allowed to bind one program while another is bound.  It is allowed to bind the
		   same program multiple times.
		   
		   Methods that bind a program will be documented.
		'''
		if not self.bound:
			GL.glUseProgram(self.handle)
		self.bound += 1
	
	def __exit__(self, ty, val, tr):
		self.bound -= 1
		if not self.bound:
			GL.glUseProgram(0)
