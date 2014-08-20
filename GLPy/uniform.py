from OpenGL import GL
from OpenGL.constants import GLboolean, GLint, GLuint, GLfloat

from itertools import product as cartesian, chain
from collections import namedtuple
import ctypes as c

import numpy

from util.misc import cumsum, product

from .datatypes import (data_types, vector_sizes, prefixes,
                        GLSLVar, BlockMember)

from .buffers import Buffer, Empty

uniform_types = { 'bool': 'i'
                , 'int': 'i'
                , 'uint': 'ui'
                , 'float': 'f' }

setter_functions = {}
getter_functions = {}
for data_type in data_types:
	code = uniform_types[data_type]
	setter_functions[data_type] = getattr(GL, 'glUniform1{}v'.format(code))
	getter_functions[data_type] = getattr(GL, 'glGetUniform{}v'.format(code))

for data_type in data_types:
	prefix = prefixes[data_type]
	code = uniform_types[data_type]
	for size in vector_sizes:
		vector_type = "{}vec{}".format(prefix, size)
		setter_functions[vector_type] = getattr(GL, 'glUniform{}{}v'.format(size, code))
		getter_functions[vector_type] = getattr(GL, 'glGetUniform{}v'.format(code))

for size1, size2 in cartesian(vector_sizes, repeat=2):
	if size1 == size2:
		f = getattr(GL, 'glUniformMatrix{}fv'.format(size1))
	else:
		f = getattr(GL, 'glUniformMatrix{}x{}fv'.format(size1, size2))
	# Numpy (and C) arrays are row-major, OpenGL expects column-major, so transpose
	# OpenGL wrapperCall doesn't work with functools.partial
	g = lambda location, count, value: f(location, count, True, value)
	matrix_type = "mat{}x{}".format(size1, size2)
	setter_functions[matrix_type] = g
	getter_functions[matrix_type] = getattr(GL, 'glGetUniformfv')
	if size1 == size2:
		matrix_type = "mat{}".format(size1)
		setter_functions[matrix_type] = g
		getter_functions[matrix_type] = getattr(GL, 'glGetUniformfv')

# TODO: Add shortcut to set multiple uniforms in program object
class UniformAttribute(GLSLVar):
	def __init__(self, program, name, gl_type, shape=1):
		super().__init__(name=name, gl_type=gl_type, shape=shape)
		self.program = program
		self.location = GL.glGetUniformLocation(self.program.handle, self.name)
	
	@classmethod
	def fromGLSLVar(cls, program, var):
		return cls(program, name=var.name, gl_type=var.type, shape=var.shape)
	
	def __str__(self):
		base = super().__str__()
		if self.location != -1:
			layout = "layout(location={})".format(self.location)
			return ' '.join((layout, base))
		return base
	
	@property
	def setter(self):
		return setter_functions[self.type]
	
	@property
	def getter(self):
		return getter_functions[self.type]
	
	@property
	def data(self):
		if self.location == -1:
			raise RuntimeError("'{}' is not a uniform attribute of {}'".format(self, self.program))
		out_buf = numpy.empty(self.shape, dtype=self.dtype)
		if self.shape == (1,):
			self.getter(self.program.handle, self.location, out_buf)
			out_buf = out_buf[0]
		else:
			for i, o in enumerate(numpy.nditer(out_buf, op_flags=['writeonly'])):
				self.getter(self.program.handle, self.location + i, o[...])
		return out_buf.T
	
	@data.setter
	def data(self, value):
		GL.glUseProgram(self.program.handle)
		self.setter(self.location, self.count, value)
		GL.glUseProgram(0)

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

# FIXME: Track layout (row/column major)
class UniformBlock:
	def __init__(self, binding, program, name, *members, instance_name=None, shape=1):
		self.name = name
		self.program = program
		self.binding = binding
		self.instance_name = instance_name or ''
		try:
			self.shape = tuple(shape)
		except TypeError:
			self.shape = (shape,)

		self.index = GL.glGetUniformBlockIndex(self.program.handle, self.name)
		GL.glUniformBlockBinding(self.program.handle, self.index, self.binding)
		self.nbytes = GL.glGetActiveUniformBlockiv(self.program.handle, self.index, GL.GL_UNIFORM_BLOCK_DATA_SIZE)

		members = [BlockMember.fromGLSLVar(self, m) for m in members]

		member_indices = glGetUniformIndices(self.program.handle, [m.api_name for m in members])
		member_offsets = glGetActiveUniformsiv(self.program.handle, member_indices, GL.GL_UNIFORM_OFFSET)

		self.members = [UniformBlockMember.fromGLSLVar(i, o, self, m) for i, o, m
		                in zip(member_indices, member_offsets, members)]

	def __str__(self):
		members = ';\n\t'.join(str(m) for m in self.members)
		"uniform {} {{\n\t{};\n}}{};".format(self.name, members, self.instance_name)
	
	@property
	def dtype(self):
		return numpy.dtype([('', m.dtype, m.shape) for m in self.members])

class UniformBlockMember(GLSLVar):
	def __init__(self, index, offset, block, name, gl_type, shape=1):
		super().__init__(name, gl_type, shape)
		self.block = block
		self.index = index
		self.offset = offset
	
	@classmethod
	def fromGLSLVar(cls, index, offset, block, var):
		return cls(index, offset, block, var.name, var.type, var.shape)

class UniformBuffer(Buffer):
	target = GL.GL_UNIFORM_BUFFER

	def __init__(self, *contents, usage=GL.GL_DYNAMIC_DRAW, handle=None):
		super().__init__(usage, handle)
		*offsets, size = list(cumsum(c.nbytes for c in contents))
		self.blocks = [UniformBlockData.fromUniformBlock(self, offset, b)
		               for b, offset in zip(contents, offsets)]

		with self:
			self.bytes[:] = Empty(size)
			for block, data in zip(contents, self.blocks):
				# ASSERT: block.nbytes == data.nbytes
				GL.glBindBufferRange(self.target, block.binding, self.handle, data.offset, block.nbytes)
	
class UniformBlockData:
	def __init__(self, buf, offset, dtype, *members):
		self.buf = buf
		self.offset = offset
		self.members = [UniformBlockMemberData.fromUniformBlockMember(self, m) for m in members]
		self.dtype = dtype
	
	@classmethod
	def fromUniformBlock(cls, buf, offset, blk):
		return cls(buf, offset, blk.dtype, *blk.members)
	
	@property
	def nbytes(self):
		return self.dtype.itemsize
	
	@property
	def data(self):
		raise NotImplementedError("Uniform Block data access not implemented.")
	
	@data.setter
	def data(self, value):
		# Members can all be different dtypes so no way to cast sensibly
		value.dtype = self.dtype
		value.shape = self.shape

		end = self.offset + self.nbytes
		self.buf[self.offset:end] = value

class UniformBlockMemberData:
	def __init__(self, blk, offset, dtype, shape=1):
		self.block = blk
		self.member_offset = offset
		self.dtype = dtype
		try:
			self.shape = tuple(shape)
		except TypeError:
			self.shape = (shape,)
	
	@classmethod
	def fromUniformBlockMember(cls, blk, member):
		return cls(blk, member.offset, member.dtype, member.shape)

	@property
	def data(self):
		raise NotImplementedError("Uniform Block data access not implemented.")
	
	@property
	def nbytes(self):
		return self.dtype.itemsize
	
	@property
	def offset(self):
		return self.member_offset + self.block.offset
	
	@data.setter
	def data(self, value):
		value = numpy.asarray(value)
		if len(value.dtype) == 0:
			value = value.astype(self.dtype.base)
			value.shape = tuple(chain(self.shape, (-1,)))
			# UPSTREAM: self.dtype doesn't work
			value.dtype = numpy.dtype([('', self.dtype)])
		else:
			value.dtype = numpy.dtype([('', self.dtype)])
			value.shape = self.shape

		end = self.offset + self.nbytes
		self.block.buf.bytes[self.offset:end] = value
