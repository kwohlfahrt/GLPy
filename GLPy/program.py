from OpenGL import GL

from .vertex import VertexAttribute

from copy import deepcopy

gl_shader_types = { 'vertex': GL.GL_VERTEX_SHADER
                  , 'fragment': GL.GL_FRAGMENT_SHADER
                  , 'geometry': GL.GL_GEOMETRY_SHADER
                  , 'tesselation control': GL.GL_TESS_CONTROL_SHADER
                  , 'tesselation evaluation': GL.GL_TESS_EVALUATION_SHADER }

#TODO: Allow re-setting sources
class Program:
	"""An OpenGL program.
	
	:param dict sources: The shader sources, where the key is the name of the shader stage.
	:param attributes: The vertex attributes used in the program
	:type attributes: [:py:class:`.VertexAttribute`]
	:param uniforms: The program's uniform attributes
	:type uniforms: [:py:class:`.UniformAttribute`]
	:param uniform_blocks: The uniform blocks defined in the program
	:type uniform_blocks: [:py:class:`.UniformBlock`]
	"""

	def __init__(self, sources, vertex_attributes=None, uniform_blocks=None):
		self.vertex_attributes = { v.name: v for v in vertex_attributes or [] }
		self.uniform_blocks = deepcopy(uniform_blocks) or []

		self.bound = 0
		self.handle = GL.glCreateProgram()
		if self.handle == 0:
			raise RuntimeError("Failed to create OpenGL program.")

		for shader_type, src in sources.items():
			gl_shader_type = gl_shader_types[shader_type]
			shader = GL.glCreateShader(gl_shader_type)
			GL.glShaderSource(shader, src)
			GL.glCompileShader(shader) 
			if GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS) == GL.GL_FALSE:
				log = GL.glGetShaderInfoLog(shader).decode()
				raise RuntimeError("Failed to compile {} shader: \n\n{}".format(shader_type, log))
			GL.glAttachShader(self.handle, shader)

		self.link()

	# TODO: Re-linking shaders not considered idiomatic in modern OpenGL, so
	# get rid of this as seperate from __init__
	def link(self):
		GL.glLinkProgram(self.handle)
		if GL.glGetProgramiv(self.handle, GL.GL_LINK_STATUS) == GL.GL_FALSE:
			log = GL.glGetProgramInfoLog(self.handle).decode()
			raise RuntimeError("Failed to link program: \n\n{}".format(log))

		for attribute in self.vertex_attributes.values():
			attribute.program = self

		# FIXME: Delete shaders when complete
		
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
