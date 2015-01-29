from OpenGL import GL

from itertools import islice
from collections import Counter, namedtuple

from .datatypes import Variable, Type
from .datatypes import Scalar, Vector, Matrix
from .buffers import Buffer, numpy_buffer_types, buffer_numpy_types, integer_buffer_types

from util.misc import product, subIter

import numpy

from ctypes import c_void_p

# These types are valid OpenGL data types for glDrawElements
gl_element_buffer_types = { GL.GL_UNSIGNED_BYTE, GL.GL_UNSIGNED_SHORT, GL.GL_UNSIGNED_INT }
element_buffer_dtypes = { buffer_numpy_types[gl_type] for gl_type in gl_element_buffer_types }

# TODO: Allow specifying attribute locations in advance
class VAO:
	'''A class to represent VAO objects.

	:param \\*attributes: The attributes that the VAO will contain.
		Locations will be automatically generated.
	:type \\*attributes: :py:class:`.VertexAttribute`
	:param handle: The OpenGL handle to use. One will be created if it is :py:obj:`None`
	:type param: :py:obj`int` or :py:obj:`None`
	'''

	def __init__(self, *attributes, handle=None):
		self.handle = GL.glGenVertexArrays(1) if handle is None else handle
		self.attributes = attributes
		self._element_buffer = None

		self.bound = 0
		with self:
			for attribute in self.attributes:
				#FIXME: Does this need to be called for every index in a matrix/array type?
				GL.glEnableVertexAttribArray(attribute.location)
	
	@property
	def element_buffer(self):
		'''The element buffer to be used with this VAO for indexed drawing.

		.. admonition:: |buffer-bind|

		   Setting to this property binds the buffer being assigned to the VAO

		.. admonition:: |vao-bind|

		   Setting to this property binds the VAO being assigned to
		'''
		return self._element_buffer
	
	@element_buffer.setter
	def element_buffer(self, value):
		if value.dtype not in element_buffer_dtypes:
			raise ValueError("Invalid dtype for an element buffer")
		with self:
			GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, value.buffer.handle)
		GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)
		self._element_buffer = value
	
	def __getitem__(self, i):
		return self.attributes[i]
	
	def __iter__(self):
		return iter(self.attributes)

	def __enter__(self):
		'''VAO objects provide a context manager. This keeps track of how many times the VAO has
		been bound and unbound. Grouping operations on a VAO within a context where it is bound
		prevents repeated binding and un-binding.

		.. _vao-bind-warning:
		.. warning::
		   It is not allowed to bind two VAOs simultaneously. It is allowed to bind the *same* VAO
		   multiple times. Methods that bind a VAO will be documented
		'''

		if not self.bound:
			GL.glBindVertexArray(self.handle)
		self.bound += 1
	
	def __exit__(self, ex, val, tr):
		self.bound -= 1
		if not self.bound:
			GL.glBindVertexArray(0)

VertexFormat = namedtuple('VertexFormat', ['components', 'buffer_format', 'offset'])

class VertexAttribute(Variable):
	"""A vertex attribute.

	:param location: The location of this vertex attribute. Will be queried
	  upon linking if not defined.
	:type location: :py:obj:`int` or :py:obj:`None`
	:raises TypeError: If the passed GLSL type is not a :py:class:`.Matrix`,
	  :py:class:`.Vector` or :py:class:`.Scalar`
	"""

	# How does glVertexAttribDivisor work with glVertexBindingDivisor?
	def __init__(self, name, gl_type, shape=1, location=None, normalized=True):
		super().__init__(name=name, gl_type=gl_type, shape=shape)
		# Could also check for base_type attribute
		if type(self.type) not in {Scalar, Vector, Matrix}:
			raise TypeError("Invalid type for a vertex attribute.")
		if location is not None:
			self.location = location
			self.location_source = 'shader'
		else:
			self.location = None
			self.location_source = 'dynamic'
		self._program = None
		self._binding = None
		self.normalized = normalized
	
	@classmethod
	def fromVariable(cls, var, location=None):
		raise NotImplementedError("TODO")

	def __str__(self):
		base = Variable.__str__(self)
		if self.location is not None:
			layout = "layout(location={})".format(self.location)
			return ' '.join((layout, 'in', base))
		return ' '.join(('in', base))

	#TODO: def __hash__(self):

	gl_type_indices = { s: 1 for s in Scalar }
	gl_type_indices.update({ v: 1 for v in Vector })
	gl_type_indices.update({ m: m.shape[0] for m in Matrix })

	@property
	def type_indices(self):
		'''The number of vertex attribute indices that would be taken up by the attribute if it
		consisted of only one array element
		'''
		return gl_type_indices[self.type]
	
	@property
	def indices(self):
		'''The total number of vertex attribute indices taken up by the attribute.'''
		return self.type_indices * product(self.shape)

	@property
	def program(self):
		'''The program this object is part of. Setting to this value will query
		   the location if it is not explicitly specified.
		'''
		return self._program

	@program.setter
	def program(self, program):
		self._program = program
		if self.location_source == 'dynamic':
			self.location = GL.glGetAttribLocation(program.handle, self.name)

	@property
	def buffer(self):
		return self._buffer

	@buffer.setter
	def buffer(self, buffer):
		if self._buffer is not None:
			self._buffer.dependents.remove(self)
		self._buffer = buffer
		buffer.dependents.add(self)

	@property
	def binding(self):
		return self._binding

	@binding.setter
	def binding(self, binding):
		if self._binding is not None:
			self._binding.attributes.remove(self)
		self._binding = binding
		self._binding.attributes.add(self)

	def dtype_specification(self, dtype):
		# TODO: Allow for more complex dtypes - one field per attribute index
		if dtype.fields is not None:
			raise ValueError("Vertex attributes must be backed by sub-array dtypes.")
		if dtype.base not in numpy_buffer_types:
			raise ValueError("{} is not a valid OpenGL data type.")

		if isinstance(self.type, Scalar):
			indices = product(dtype.shape)
			components = 1
		else:
			indices = product(dtype.shape[:-1])
			components = dtype.shape[-1]
		
		if not 1 <= components <= 4:
			raise ValueError("Data type specifies invalid number of components: {}"
			                 .format(components))
		if indices < self.indices:
			raise ValueError("Data type specifies {} attributes, need {}."
			                 .format(columns, self.type.shape[1]))

		index_offsets = [dtype.base.itemsize * i for i in range(indices)]
		buffer_formats = repeat(numpy_buffer_types[dtype.base], self.indices)
		components = repeat(components, self.indices)
		
		return [VertexFormat(c, f, o) for c, f, o in
		        zip(components, buffer_formats, index_offsets) ]

	# TODO: Make assignment to self.binding handle these?
	def set_binding(self, binding, index=0, item=0):
		item_dtype = binding.buffer_block.dtype[index]
		relative_offset = ( sum(binding.buffer_block.dtype[i].itemsize for i in range(index))
		                  + (item * binding.buffer_block.dtype.itemsize) )
		
		vertex_formats = self.dtype_specificiation(item_dtype)

		for index, format in enumerate(vertex_formats):
			glVertexAttribFormat(self.location + index, format.components
			                    , format.buffer_format, normalized, format.offset)
		self.binding = binding

class VertexAttributeBinding:
	'''A class representing a vertex attribute binding point.
	
	:param int index: The OpenGL binding index. Must be unique and less than
	  ``GL_MAX_VERTEX_ATTRIB_BINDINGS``.
	:param int divisor: The divisor for this vertex binding.
	'''
	def __init__(self, index, divisor=0):
		self.index = index
		self._buffer_block = None
		self.vertex_attributes = set()
		self.divisor = divisor

	@property
	def buffer_block(self):
		return self._buffer_block

	@buffer_block.setter
	def buffer_block(self, buffer_block):
		if self._buffer_block is not None:
			self._buffer_block.dependents.remove(self)
		self._buffer_block = buffer_block
		self._buffer_block.dependents.add(self)

	@property
	def divisor(self):
		return self._divisor

	@divisor.setter
	def divisor(self, value):
		GL.glVertexBindingDivisor(self.index, value)
		self._divisor = value

	# TODO: Make assignment to self.buffer_block handle these?
	def bind_buffer_block(self, buffer_block, offset=0):
		offset = buffer_block.offset + offset * buffer_block.stride
		GL.glBindVertexBuffer( self.index, buffer_block.buffer.handle
		                     , offset, buffer_block.stride)
		self.buffer_block = buffer_block
