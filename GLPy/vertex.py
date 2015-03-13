from OpenGL import GL

from itertools import repeat, chain
from collections import Counter, namedtuple

from .GLSL import Variable, Scalar, Vector, Matrix, BasicType, Array
from .buffers import Buffer, numpy_buffer_types, buffer_numpy_types, integer_buffer_types

from util.misc import product, subIter

import numpy

from ctypes import c_void_p

class VertexAttribute(Variable):
	"""A vertex attribute.

	:param location: The location of this vertex attribute. Will be queried
	  upon linking if not defined.
	:type location: :py:obj:`int` or :py:obj:`None`
	:raises TypeError: If the passed GLSL type is not an instance of :py:class:`.BasicType` or an
	  :py:class:`.Array` of :py:class:`.BasicType`
	"""

	program = None #: The program containing this attribute

	# How does glVertexAttribDivisor work with glVertexBindingDivisor?
	def __init__(self, name, datatype, shader_location=None, normalized=True):
		super().__init__(name=name, datatype=datatype)
		# Could also check for base_type attribute
		base_type = getattr(self.datatype, 'base', self.datatype)
		if not isinstance(base_type, BasicType):
			raise TypeError("Invalid type for a vertex attribute.")
		self.shader_location = shader_location
		'''The location of this attribute if specified in the shader'''
		self.normalized = normalized 
		'''Whether a floating-point data type should be normalized from integer data'''
		self.VAOs = []
		'''The VAOs that contain this attribute'''
	
	@classmethod
	def fromVariable(cls, var, shader_location=None, normalized=True):
		return cls(var.name, var.datatype, shader_location, normalized)

	def __str__(self):
		base = super().__str__()
		if self.shader_location is not None:
			layout = "layout(location={})".format(self.location)
			return ' '.join((layout, 'in', base))
		return ' '.join(('in', base))

	type_indices = { s: 1 for s in Scalar }
	type_indices.update({ v: 1 for v in Vector })
	type_indices.update({ m: m.shape[0] for m in Matrix })

	@property
	def indices(self):
		'''The total number of vertex attribute indices taken up by the attribute.'''
		datatype = getattr(self.datatype, 'base', self.datatype)
		element_indices = getattr(datatype, 'columns', 1)
		array_shape = getattr(self.datatype, 'full_shape', (1,))
		return element_indices * product(array_shape)

	@property
	def dynamic_location(self):
		return GL.glGetAttribLocation(self.program.handle, self.name)

	@property
	def location(self):
		if self.shader_location is not None:
			return self.shader_location
		return self.dynamic_location
	
	@property
	def locations(self):
		return range(self.location, self.location + self.indices)
	
	@property
	def components(self):
		'''The maximum number of components of a single attribute index of this type.'''

		datatype = getattr(self.datatype, 'base', self.datatype)
		return getattr(datatype, 'shape', (1,))[-1]

# These types are valid OpenGL data types for glDrawElements
gl_element_buffer_types = { GL.GL_UNSIGNED_BYTE, GL.GL_UNSIGNED_SHORT, GL.GL_UNSIGNED_INT }
element_buffer_dtypes = { buffer_numpy_types[datatype] for datatype in gl_element_buffer_types }

class VAO:
	'''A class to represent VAO objects.

	:param \\*attributes: The attributes that the VAO will contain.
	:type \\*attributes: :py:class:`.VertexAttribute`
	:param handle: The OpenGL handle to use. One will be created if it is :py:obj:`None`
	:type param: :py:obj`int` or :py:obj:`None`
	:raises GL.GLError: If any attribute attempts to define locations greater than
	  :py:obj:`GL.GL_MAX_VERTEX_ATTRIBS`
	'''

	_element_buffer = None

	def __init__(self, *attributes, handle=None):
		self.handle = GL.glGenVertexArrays(1) if handle is None else handle
		self.attributes = {}
		self.bound = 0

		vao_attributes = chain.from_iterable(VAOAttribute.fromVertexAttribute(self, a)
		                                     for a in attributes)

		for attribute in vao_attributes:
			self.attributes[attribute.location] = attribute

		with self:
			for location in self.attributes:
				GL.glEnableVertexAttribArray(location)
	
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
		if value.dtype.base not in element_buffer_dtypes:
			raise ValueError("Invalid dtype for an element buffer")
		with self:
			GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, value.buffer.handle)
		GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)
		self._element_buffer = value

	def __getitem__(self, i):
		return self.attributes[i]

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
	
# So they take the same number of paremeters as GL.glVertexAttribPointer
def glVertexAttribIPointer(idx, components, type, normalized, stride, offset):
	GL.glVertexAttribIPointer(idx, components, type, stride, offset)
def glVertexAttribLPointer(idx, components, type, normalized, stride, offset):
	GL.glVertexAttribLPointer(idx, components, type, stride, offset)

class VAOAttribute:
	'''An attribute that is specified in a VAO.

	:param vao: The VAO this attribute belongs to.
	:type vao: :py:class:`.VAO`
	:param int location: The location (index) of this attribute in the VAO
	:param scalar_type: The scalar type of one element of this attribute
	:type scalar_type: :py:class:`.Scalar`
	:param bool normalized: If the attribute is of a floating-point type,
	  whether data should be normalized
	'''

	gl_pointer_functions = { Scalar.float: GL.glVertexAttribPointer 
	                       , Scalar.double: glVertexAttribLPointer
	                       , Scalar.uint: glVertexAttribIPointer
	                       , Scalar.bool: GL.glVertexAttribPointer
	                       , Scalar.int: glVertexAttribIPointer }

	def __init__(self, vao, location, components, scalar_type, divisor=0, normalized=True):
		self.vao = vao
		self.location = location
		self.scalar_type = scalar_type
		self.components = components
		self.divisor = divisor
		self.normalized = bool(normalized)
		self._data = None
	
	@classmethod
	def fromVertexAttribute(cls, vao, attribute, divisor=0):
		scalar_type = getattr(attribute.datatype, 'base', attribute.datatype).scalar_type
		return [cls(vao, l, attribute.components, scalar_type, divisor, attribute.normalized)
		        for l in attribute.locations]
	
	@property
	def data(self):
		'''The buffer data backing this vertex attribute. Currently set-only.

		:param value: The data for the attribute.
		:type value: :py:class:`.BufferItem`
		'''
		return self._data
	
	@data.setter
	def data(self, value):
		if value.components < self.components:
			# Never allow, GL_ARB_vertex_attrib_64bit suggests default values for compatability
			raise ValueError("Specified only {} components for a vertex attribute expecting {}."
			                 .format(value.components, self.components))
		with value.buffer, self.vao:
			setter = self.gl_pointer_functions[self.scalar_type]
			setter(self.location, value.components, numpy_buffer_types[value.dtype.base],
			       self.normalized, value.buffer.stride, GL.GLvoidp(value.offset))
	@property
	def divisor(self):
		return self._divisor
	
	@divisor.setter
	def divisor(self, value):
		'''The divisor for this vertex attribute. If it is 0, one element of the attribute will be
		consumed per vertex rendered. If it is non-zero, one element will be consumed for each
		*divisor* instances of the shader invoked.

		:param int value: The number of instances to render using each value of this attribute, or
		  ``0`` to use one value per vertex.
		'''
		with self.vao:
			GL.glVertexAttribDivisor(self.location, int(value))
		self._divisor = value
