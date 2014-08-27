from OpenGL import GL
from OpenGL.constants import GLboolean, GLint, GLuint, GLfloat

from itertools import product as cartesian, chain, accumulate
from collections import namedtuple
import ctypes as c

import numpy

from util.misc import product, isContiguous

from .datatypes import (data_types, sampler_types, vector_sizes, prefixes,
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

for sampler_type in sampler_types.keys():
	for data_type in ['int', 'uint', 'float']:
		gl_type = "{}sampler{}".format(prefixes[data_type], sampler_type)
		setter_functions[gl_type] = getattr(GL, 'glUniform1iv'.format(code))
		getter_functions[gl_type] = getattr(GL, 'glUniform1iv'.format(code))

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
	'''A uniform attribute bound to a program

	:param program: The program the attribute is bound to.
	:type program: :py:class:`.Program`
	'''
	def __init__(self, program, name, gl_type, shape=1):
		super().__init__(name=name, gl_type=gl_type, shape=shape)
		self.program = program
		self.location = GL.glGetUniformLocation(self.program.handle, self.name)
	
	@classmethod
	def fromGLSLVar(cls, program, var):
		'''Construct from a :py:class:`.Program` and :py:class:`.GLSLVar`

		:param var: The variable the attribute is based on
		:type var: :py:class:`.GLSLVar`
		:param program: The program containing the attribute
		:type program: :py:class:`.Program`
		'''
		return cls(program, name=var.name, gl_type=var.type, shape=var.shape)
	
	def __str__(self):
		base = super().__str__()
		if self.location != -1:
			layout = "layout(location={})".format(self.location)
			return ' '.join((layout, base))
		return base
	
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

# TODO: Track layout (row/column major)
class UniformBlock:
	''' '''
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
		return "uniform {} {{\n\t{};\n}}{};".format(self.name, members, self.instance_name)
	
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
	'''A buffer containing uniform data. May contain data for one or more uniform blocks.

	:param contents: The uniform blocks backed by this buffer
	:type contents: [:py:class:`.UniformBlock`]
	'''

	# UPSTREAM: Docstring currently invisible due to Sphinx Issue #1547
	target = GL.GL_UNIFORM_BUFFER
	'''This buffer binds GL.GL_UNIFORM_BUFFER'''

	def __init__(self, *contents, usage=GL.GL_DYNAMIC_DRAW, handle=None):
		super().__init__(usage, handle)
		*offsets, size = chain([0], accumulate(c.nbytes for c in contents))
		self.blocks = [UniformBlockData.fromUniformBlock(self, offset, b)
		               for b, offset in zip(contents, offsets)]

		with self:
			self.bytes[...] = Empty(size)
			for block, data in zip(contents, self.blocks):
				GL.glBindBufferRange(self.target, block.binding, self.handle, data.offset, block.nbytes)
	
	#TODO: Add __setitem__

class UniformBlockData:
	'''A class to represent the storage of one block in a UniformBuffer

	:param buf: The buffer containing the block
	:type buf: :py:class:`.UniformBuffer`
	:param int offset: The offset of the block within the buffer
	:param members: The members of the block
	:type members: :py:class:`.BlockMember`
	'''
	def __init__(self, buf, offset, *members, shape=1):
		self.buf = buf
		self.offset = offset
		self.members = [UniformBlockMemberData.fromUniformBlockMember(self, m) for m in members]
		self.shape = shape
	
	@classmethod
	def fromUniformBlock(cls, buf, offset, blk):
		'''Construct from a :py:class:`.UniformBlock`

		:param buf: The buffer containing the block
		:type buf: :py:class:`.UniformBuffer`
		:param int offset: The offset of the block within the buffer
		:param block: The uniform block to be stored
		:type block: :py:class:`.UniformBlock`
		'''
		return cls(buf, offset, *blk.members, shape=blk.shape)

	@property
	def dtype(self):
		'''The dtype of one element in the uniform block'''
		return numpy.dtype([('', m.dtype, m.shape) for m in self.members])
	
	@property
	def nbytes(self):
		'''The total number of bytes required to store all elements of this uniform block'''
		return self.dtype.itemsize * product(self.shape)
	
	def __getitem__(self, idx):
		raise NotImplementedError("Uniform Block data access not implemented.")
	
	def __setitem__(self, idxs, value):
		'''Set components of the uniform block

		:param idxs: The section of the block to be set. Last index refers to the members of the
			block, all others refer to blocks within an array of blocks
		:param value: The new data for the selected block members
		:raises IndexError: On attempts to set non-contiguous buffer sections
		'''
		value = numpy.asarray(value)

		if all(s == 1 for s in self.shape):
			idx = idxs if isinstance(idxs, slice) else slice(idxs, idxs + 1)
			idxs = [slice(1), idx]
		else:
			idxs = (i if isinstance(i, slice) else slice(i, i + 1) for i in idxs)
		if not isContiguous(idxs, self.shape):
			raise IndexError("Only contiguous buffer sections may be set.")

		*elements, members = idxs
		members = range(*members.indices(len(self.members)))
		elements = [range(*e.indices(s)) for e, s in zip(elements, self.shape)]

		value = value.ravel()
		value.dtype = numpy.dtype([('', self.dtype.fields[n][0].base, self.dtype.fields[n][0].shape)
								   for n in self.dtype.names[members.start:members.stop]])
		value.shape = [len(r) for r in elements]

		start = ( self.offset
		        + product(e.start for e in elements) * self.dtype.itemsize 
		        + sum(m.dtype.itemsize for m in self.members[:members.start]))
		end = ( start
		      + sum(m.dtype.itemsize for m in self.members[members.start:members.stop])
			  * product(e.stop - e.start for e in elements))
		self.buf.bytes[start:end] = value

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
	def nbytes(self):
		return self.dtype.itemsize

	# TODO: Add __setitem__
