from OpenGL import GL
from OpenGL.constants import GLboolean, GLint, GLuint, GLfloat, GLdouble

from itertools import product as cartesian, chain, accumulate
from functools import partial
from collections import namedtuple
import ctypes as c

import numpy
from numpy import dtype, nditer

from util.misc import product

from .datatypes import ( Scalar, Vector, Matrix, Sampler, Variable
                       , InterfaceBlock, InterfaceBlockMember )

from .buffers import Buffer

uniform_numpy_types = { Scalar.bool: dtype(GLint)
                      , Scalar.int: dtype(GLint)
                      , Scalar.uint: dtype(GLuint)
                      , Scalar.float: dtype(GLfloat)
                      , Scalar.double: dtype(GLdouble) }
uniform_numpy_types.update({ v: dtype((uniform_numpy_types[v.base_type], v.shape[0]))
                              for v in Vector })
uniform_numpy_types.update({ m: dtype((uniform_numpy_types[m.base_type], m.shape))
                              for m in Matrix })
uniform_numpy_types.update({ s: dtype(GLint) for s in Sampler})
# UNIFORM_MATRIX_STRIDE between matrix rows/columns
# UNIFORM_ARRAY_STRIDE for arrays

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
	code = uniform_codes[vector.base_type.name]
	setter_functions[vector] = getattr(GL, 'glUniform{}{}v'.format(vector.shape[0], code))
	getter_functions[vector] = getattr(GL, 'glGetUniform{}v'.format(code))

for sampler in Sampler:
		setter_functions[sampler] = GL.glUniform1iv
		getter_functions[sampler] = GL.glGetUniformiv

for matrix in Matrix:
	code = uniform_codes[matrix.base_type.name]
	if matrix.shape[0] == matrix.shape[1]:
		f = getattr(GL, "glUniformMatrix{}{}v".format(matrix.shape[0], code))
	else:
		f = getattr(GL, "glUniformMatrix{}x{}{}v".format(matrix.shape[0], matrix.shape[1], code))
	# OpenGL wrapperCall doesn't work with functools.partial
	# Double 'lambda' to make sure 'f' is bound to the correct function
	g = (lambda f: lambda location, count, value: f(location, count, True, value))(f)
	setter_functions[matrix] = g
	getter_functions[matrix] = getattr(GL, "glGetUniform{}v".format(code))

# TODO: Uniform structs
class UniformAttribute(Variable):
	'''A uniform attribute bound to a program.
	
	:param location: The location of the uniform variable, if defined in the shader source
	:type location: :py:obj:`int` or :py:obj:`None`
	'''
	def __init__(self, name, gl_type, shape=1, location=None):
		super().__init__(name=name, gl_type=gl_type, shape=shape)
		self._program = None
		if location is not None:
			self.location = location
			self.location_source = 'shader'
		else:
			self.location = None
			self.location_source = 'dynamic'
	
	@classmethod
	def fromGLSLVar(cls, var, location=None):
		'''Construct from a :py:class:`.Program` and :py:class:`.Variable`'''
		return cls(name=var.name, gl_type=var.type, shape=var.shape, location=location)
	
	def __str__(self):
		base = super().__str__()
		if self.location is not None:
			layout = "layout(location={})".format(self.location)
			return ' '.join((layout, 'uniform', base))
		return ' '.join(('uniform', base))

	@property
	def program(self):
		return self._program

	@program.setter
	def program(self, program):
		self._program = program
		if self.location_source == 'dynamic':
			self.location = GL.glGetUniformLocation(program.handle, self.name)

	@property
	def dtype(self):
		return uniform_numpy_types[self.type]
	
	@property
	def setter(self):
		'''The OpenGL function used to set a uniform attribute of this type'''
		return setter_functions[self.type]
	
	@property
	def getter(self):
		'''The OpenGL function used to read a uniform attribute of this type'''
		return getter_functions[self.type]
	
	@property
	def data(self):
		'''A property for access to the value of the attribute.

		:raises RuntimeError: On attempting to *read* an attribute that does not exist within the
			program (i.e. has a location of ``-1``). *Writing* to such an attribute is allowed (but
			will have no effect).

		.. admonition |program-bind|

		   Setting a uniform attribute binds the program that contains it.
		'''
		if self.location is None:
			raise RuntimeError("{} has not yet been bound to a program".format(self))
		if self.location == -1:
			raise RuntimeError("'{}' is not a uniform attribute of {}'".format(self, self.program))
		out_buf = numpy.empty(self.shape, dtype=self.dtype)
		if self.shape == (1,):
			self.getter(self.program.handle, self.location, out_buf)
			out_buf = out_buf[0]
		else:
			for i, o in enumerate(nditer(out_buf, op_flags=['writeonly'])):
				self.getter(self.program.handle, self.location + i, o[...])
		return out_buf.T
	
	@data.setter
	def data(self, value):
		# Refactor into property? And raise error if accessed before binding?
		if self.location is None:
			raise RuntimeError("{} has not yet been bound to a program".format(self))
		with self.program:
			self.setter(self.location, self.count, value)

def glGetUniformIndices(program, uniform_names):
	name_array = c.c_char_p * len(uniform_names)
	c_uniform_names = name_array(*[c.c_char_p(name.encode()) for name in uniform_names])
	c_uniform_names = c.cast(c_uniform_names, c.POINTER(c.POINTER(c.c_char)))

	uniform_indices = numpy.empty(len(uniform_names), dtype=GLuint)
	GL.glGetUniformIndices(program, len(uniform_names), c_uniform_names, uniform_indices)
	return uniform_indices

def glGetActiveUniformsiv(program, indices, pname):
	# UPSTREAM: Using a numpy array here doesn't work for some reason
	out_array = (GLuint * len(indices))()
	GL.glGetActiveUniformsiv(program, len(indices), indices, pname, out_array)
	return [i for i in out_array]

# CONTINUE HERE: Leave program, locations, etc uninitialized,
# then initialize on binding to a program. Make sure to take proper steps for packed vs shared
# Test glGetProgramResource*
class UniformBlock(InterfaceBlock):
	'''An OpenGL Uniform Block.'''
	def __init__(self, program, binding, name, *members, instance_name='', shape=1, layout='shared'):
		super().__init__(name, *members, instance_name=instance_name, shape=shape, layout=layout)
		self.program = program
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
		return numpy.dtype([('', m.dtype, m.shape) for m in self.members])

# Should not be instantiated directly
class UniformBlockMember(InterfaceBlockMember):
	def __init__(self, index, offset, block, name, gl_type, shape=1):
		super().__init__(block, name, gl_type, shape)
		self.index = index
		self.offset = offset
	
	@classmethod
	def fromBlockMember(cls, index, offset, var):
		return cls(index, offset, var.block, var.name, var.type, var.shape)

	# Shaered with UniformAttribute - common to uniform variables
	@property
	def dtype(self):
		return uniform_numpy_types[self.type]

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
