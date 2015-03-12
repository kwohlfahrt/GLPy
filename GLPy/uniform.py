from OpenGL import GL
from OpenGL.constants import GLboolean, GLint, GLuint, GLfloat, GLdouble

from itertools import product as cartesian, chain, accumulate, repeat
import ctypes as c

from numpy import dtype, nditer, empty

from util.misc import product

from .GLSL import ( Scalar, Vector, Matrix, Sampler, Variable, BasicType
                  , Struct, Array, InterfaceBlock, InterfaceBlockMember )

from .buffers import Buffer

uniform_codes = { 'bool': 'i'
                , 'int': 'i'
                , 'uint': 'ui'
                , 'float': 'f'
                , 'double': 'd' }

setter_functions = {}
getter_functions = {}
for scalar in Scalar:
	code = uniform_codes[scalar.name]
	setter_functions[scalar] = getattr(GL, 'glUniform1{}v'.format(code))
	getter_functions[scalar] = getattr(GL, 'glGetUniform{}v'.format(code))

for vector in Vector:
	code = uniform_codes[vector.scalar_type.name]
	setter_functions[vector] = getattr(GL, 'glUniform{}{}v'.format(vector.shape[0], code))
	getter_functions[vector] = getattr(GL, 'glGetUniform{}v'.format(code))

for sampler in Sampler:
		setter_functions[sampler] = GL.glUniform1iv
		getter_functions[sampler] = GL.glGetUniformiv

for matrix in Matrix:
	code = uniform_codes[matrix.scalar_type.name]
	if matrix.shape[0] == matrix.shape[1]:
		f = getattr(GL, "glUniformMatrix{}{}v".format(matrix.shape[0], code))
	else:
		f = getattr(GL, "glUniformMatrix{}x{}{}v".format(matrix.shape[0], matrix.shape[1], code))
	# OpenGL wrapperCall doesn't work with functools.partial
	# Double 'lambda' to make sure 'f' is bound to the correct function
	g = (lambda f: lambda location, count, value: f(location, count, True, value))(f)
	setter_functions[matrix] = g
	getter_functions[matrix] = getattr(GL, "glGetUniform{}v".format(code))

class Uniform(Variable):
	'''A uniform attribute that may be bound to a program.
	
	:param location: The location of the uniform variable, if defined in the shader source.
	:type location: :py:obj:`int` or :py:obj:`None`
	'''

	def __init__(self, name, gl_type, location=None):
		super().__init__(name=name, gl_type=gl_type)
		self._program = None
		self.shader_location = location
		self.dynamic_location = None

	@classmethod
	def fromVariable(cls, var, location=None):
		'''Construct from a :py:class:`.Variable` and an optional location.

		:returns: The uniform attributes defined by a variable
		:rtype: [:py:class:`UniformAttribute`]
		'''
		return cls(var.name, var.type, location)

	@property
	def resources(self):
		resources = super().resources
		if self.shader_location is None:
			locations = repeat(None, len(resources))
		else:
			sizes = (getattr(r, 'array_shape', 1) for r in resources)
			locations = accumulate(chain((self.shader_location,), sizes))
		return [Uniform.fromVariable(r, l) for r, l in zip(resources, locations)]

	def __str__(self):
		base = super().__str__()
		if self.shader_location is not None:
			layout = "layout(location={})".format(self.shader_location)
			return ' '.join((layout, 'uniform', base))
		return ' '.join(('uniform', base))

	@property
	def program(self):
		return self._program

	@program.setter
	def program(self, program):
		self._program = program
		if self.shader_location is None:
			self.dynamic_location = GL.glGetUniformLocation(program.handle, self.name)

	@property
	def location(self):
		if self.shader_location is None:
			return self.dynamic_location
		else:
			return self.shader_location

	@property
	def dtype(self):
		if isinstance(self.type, Array):
			return dtype((self.type.base.machine_type, self.type.array_shape))
		else:
			return self.type.machine_type

	@property
	def setter(self):
		'''The OpenGL function used to set a uniform attribute of this type'''
		try:
			return setter_functions[self.type]
		except KeyError:
			return setter_functions[self.type.element]

	@property
	def getter(self):
		'''The OpenGL function used to read a uniform attribute of this type'''
		try:
			return getter_functions[self.type]
		except KeyError:
			return getter_functions[self.type.element]

	@property
	def data(self):
		'''A property for access to the value of the attribute.

		:raises RuntimeError: On attempting to *read* an attribute that does not exist within the
			program (i.e. has a location of ``-1``). *Writing* to such an attribute is allowed (but
			will have no effect).

		.. admonition |program-bind|

		   *Setting* a uniform attribute binds the program that contains it.
		'''
		if self.location is None:
			raise RuntimeError("{} has not yet been bound to a program".format(self))
		if self.location == -1:
			raise RuntimeError("'{}' is not a uniform attribute of {}'".format(self, self.program))

		dtype, shape = self.dtype, ()
		if dtype.base.subdtype is not None:
			dtype, shape = dtype.subdtype

		out_buf = empty(shape, dtype=dtype)
		if shape is ():
			self.getter(self.program.handle, self.location, out_buf)
		else:
			for idx, row in enumerate(out_buf):
				self.getter(self.program.handle, self.location + idx, row[...])
		return out_buf

	@data.setter
	def data(self, value):
		if self.location is None:
			raise RuntimeError("{} has not yet been bound to a program".format(self))

		dtype, shape = self.dtype, ()
		if dtype.base.subdtype is not None:
			dtype, shape = dtype.subdtype
		value.shape = shape + dtype.shape
		if value.shape != shape + dtype.shape:
			raise ValueError("Incorrect shape for uniform variable, expecting {}, got {}"
			                 .format(shape + dtype.shape, value.shape))
		if value.dtype != dtype.base:
			raise ValueError("Incorrect data type for uniform variable, expecting {}, got {}"
			                 .format(dtype.base, value.dtype))

		count = getattr(self.type, 'array_shape', 1)
		with self.program:
			self.setter(self.location, count, value)

def glGetUniformIndices(program, uniform_names):
	name_array = c.c_char_p * len(uniform_names)
	c_uniform_names = name_array(*[c.c_char_p(name.encode()) for name in uniform_names])
	c_uniform_names = c.cast(c_uniform_names, c.POINTER(c.POINTER(c.c_char)))

	uniform_indices = empty(len(uniform_names), dtype=GLuint)
	GL.glGetUniformIndices(program, len(uniform_names), c_uniform_names, uniform_indices)
	return uniform_indices

def glGetActiveUniformsiv(program, indices, pname):
	# UPSTREAM: Using a numpy array here doesn't work for some reason
	out_array = (GLuint * len(indices))()
	GL.glGetActiveUniformsiv(program, len(indices), indices, pname, out_array)
	return [i for i in out_array]

# UNIFORM_MATRIX_STRIDE between matrix rows/columns
# UNIFORM_ARRAY_STRIDE for arrays

# then initialize on binding to a program. Make sure to take proper steps for packed vs shared
# Test glGetProgramResource*
class UniformBlock(InterfaceBlock):
	'''An OpenGL Uniform Block.'''
	def __init__(self, name, *members, instance_name='', layout='shared'):
		super().__init__(name, *members, instance_name=instance_name, shape=shape, layout=layout)
		self.binding = binding

		self.index = GL.glGetUniformBlockIndex(self.program.handle, self.name)
		GL.glUniformBlockBinding(self.program.handle, self.index, self.binding)
		self.nbytes = GL.glGetActiveUniformBlockiv(self.program.handle, self.index, GL.GL_UNIFORM_BLOCK_DATA_SIZE)

		members = [InterfaceBlockMember.fromVariable(self, m) for m in members]

		member_indices = glGetUniformIndices(self.program.handle, [m.api_name for m in members])
		member_offsets = glGetActiveUniformsiv(self.program.handle, member_indices, GL.GL_UNIFORM_OFFSET)

		self.members = [UniformBlockMember.fromBlockMember(i, o, m) for i, o, m
		                in zip(member_indices, member_offsets, members)]

	def __str__(self):
		members = ';\n\t'.join(str(m) for m in self.members)
		return "uniform {} {{\n\t{};\n}}{};".format(self.name, members, self.instance_name)
	
	@property
	def dtype(self):
		return dtype([('', m.dtype, m.shape) for m in self.members])

# Should not be instantiated directly
class UniformBlockMember(InterfaceBlockMember):
	def __init__(self, index, offset, block, name, gl_type, shape=1):
		super().__init__(block, name, gl_type, shape)
		self.index = index
		self.offset = offset
	
	@classmethod
	def fromBlockMember(cls, index, offset, var):
		return cls(index, offset, var.block, var.name, var.type, var.shape)

class UniformBinding:
	def __init__(self, index):
		self.index = index
		self._buffer_block = None
		self.uniform_blocks = set()

	# TODO: Shared with vertex.VertexAttribBinding, refactor
	@property
	def buffer_block(self):
		return self._buffer_block

	@buffer_block.setter
	def buffer_block(self, buffer_block):
		if self._buffer_block is not None:
			self._buffer_block.dependents.remove(self)
		self._buffer_block = buffer_block
		self._buffer_block.dependents.add(self)
		buffer_block.dependents.add(self)

	def bind_buffer_block(self, buffer_block, offset=0):
		offset = buffer_block.offset + offset * buffer_block.stride
		size = buffer_block.nbytes - offset
		with buffer_block.buffer.bind(GL.GL_UNIFORM_BUFFER):
			GL.glBindBufferRange( GL.GL_UNIFORM_BUFFER, self.index
			                    , buffer_block.buffer.handle, offset, size)
		self.buffer_block = buffer_block
