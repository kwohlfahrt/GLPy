from OpenGL import GL

from .GLSL import Scalar, BasicType, VertexAttribute
from .buffers import numpy_buffer_types, buffer_numpy_types

from util.misc import product

from ctypes import c_void_p

class ProgramVertexAttribute(VertexAttribute):
	"""A vertex attribute of a program.

	:param program: The shader program this attribute belongs to
	:type program: :py:class:`.Program`
	"""

	# How does glVertexAttribDivisor work with glVertexBindingDivisor?
	def __init__(self, program, name, datatype, location=None, normalized=True):
		self.program = program
		super().__init__(name, datatype, location, normalized)

	@classmethod
	def fromVertexAttribute(cls, program, va):
		return cls(program, va.name, va.datatype, va.shader_location, va.normalized)

	@property
	def dynamic_location(self):
		return GL.glGetAttribLocation(self.program.handle, self.name)

	@property
	def location(self):
		if self.shader_location is not None:
			return self.shader_location
		return self.dynamic_location

# These types are valid OpenGL data types for glDrawElements
gl_element_buffer_types = { GL.GL_UNSIGNED_BYTE, GL.GL_UNSIGNED_SHORT, GL.GL_UNSIGNED_INT }
element_buffer_dtypes = { buffer_numpy_types[datatype] for datatype in gl_element_buffer_types }

class VAO:
	'''A class to represent VAO objects.

    .. warning::

       The VAO will be bound during initialization.

	:param \\*attributes: The attributes that the VAO will contain.
	:type \\*attributes: :py:class:`.VertexAttribute`
	:param handle: The OpenGL handle to use. One will be created if it is :py:obj:`None`
	:type param: :py:obj`int` or :py:obj:`None`
	:raises GL.GLError: If any attribute attempts to define locations greater than
	  :py:obj:`GL.GL_MAX_VERTEX_ATTRIBS`
	'''

	def __init__(self, *attributes, handle=None):
		self.handle = GL.glGenVertexArrays(1) if handle is None else handle
		self.attributes = {a.name: VAOAttribute.fromVertexAttribute(self, a) for a in attributes}
		self._element_buffer = None

		occupied_attributes = set()
		with self:
			for attribute in self.attributes.values():
				attribute_locations = set(attribute.locations)
				if occupied_attributes & attribute_locations:
					raise ValueError("Cannot specify overlapping attributes on one VAO.")
				occupied_attributes |= attribute_locations
				for location in attribute.locations:
					GL.glEnableVertexAttribArray(location)

	@property
	def element_buffer(self):
		'''The element buffer to be used with this VAO for indexed drawing.

		.. warning:: |buffer-bind|

		   Setting to this property binds the buffer being assigned to the VAO to
		   :py:obj:`GL.GL_ELEMENT_ARRAY_BUFFER`.

		.. warning:: |vao-bind|

		   Setting to this property binds the VAO being assigned to.
		'''
		return self._element_buffer

	@element_buffer.setter
	def element_buffer(self, value):
		if value.dtype.base not in element_buffer_dtypes:
			raise ValueError("Invalid dtype for an element buffer")
		with self:
			GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, value.handle)
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

		   It is not allowed to bind two VAOs (or the same VAO twice) simultaneously.
		'''

		GL.glBindVertexArray(self.handle)

	def __exit__(self, ex, val, tr):
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
	:param scalar_type: The scalar type of this attribute
	:type scalar_type: :py:class:`.Scalar`
	:param bool normalized: If the attribute is of a floating-point type,
	  whether data should be normalized
	'''

	gl_pointer_functions = { Scalar.float: GL.glVertexAttribPointer
	                       , Scalar.double: glVertexAttribLPointer
	                       , Scalar.uint: glVertexAttribIPointer
	                       , Scalar.bool: GL.glVertexAttribPointer
	                       , Scalar.int: glVertexAttribIPointer }

	def __init__(self, vao, location, datatype, normalized=False, divisor=0):
		self.normalized = normalized
		try:
			self.datatype = BasicType(datatype)
		except ValueError:
			self.datatype = datatype
		if not isinstance(getattr(self.datatype, 'base', self.datatype), BasicType):
			raise ValueError("Vertex attributes must be basic types or arrays thereof.")
		self.datatype = datatype
		self.location = location
		self.vao = vao
		self.divisor = divisor
		self._data = None

	def __repr__(self):
		return ( "<VAOAttribute vao={}, location={}, length={}, normalized={}, divisor={}>"
		         .format(self.vao, self.location, self.locations, self.normalized, self.divisor) )

	@classmethod
	def fromVertexAttribute(cls, vao, attrib, divisor=0):
		location = getattr(attrib, 'location', attrib.shader_location)
		if location is None:
			raise ValueError("Cannot construct from attribute without location.")
		return cls(vao, location, attrib.datatype, attrib.normalized, divisor)

	def __eq__(self, other):
		# Could check scalar type, locations and components instead of datatype
		return ( self.vao == other.vao and self.location == other.location
		       and self.datatype == other.datatype and self.normalized == other.normalized
		       and self.divisor == other.divisor )

	def __getitem__(self, idx):
		datatype = self.datatype[idx]

		array_elements = product(getattr(datatype, 'full_shape', (1,)))
		base_type = getattr(datatype, 'base', datatype)
		element_indices = getattr(base_type, 'columns', 1)

		location = self.location + idx * element_indices * array_elements
		return VAOAttribute(self.vao, location, datatype, self.normalized, self.divisor)

	@property
	def indices(self):
		'''The total number of vertex attribute indices taken up by the attribute.'''

		datatype = getattr(self.datatype, 'base', self.datatype)
		element_indices = getattr(datatype, 'columns', 1)
		array_shape = getattr(self.datatype, 'full_shape', (1,))
		return element_indices * product(array_shape)

	@property
	def locations(self):
		'''All of the locations occupied by this attribute.

		:rtype: [:py:obj:`int`]
		'''
		return range(self.location, self.location + self.indices)

	@property
	def components(self):
		'''The number of components of a single attribute index of this type.'''

		datatype = getattr(self.datatype, 'base', self.datatype)
		return getattr(datatype, 'shape', (1,))[-1]

	@property
	def data(self):
		'''The buffer data backing this vertex attribute.

		.. warning::

		   Setting to this attribute binds the buffer providing the data to
		   :py:obj:`GL.GL_ARRAY_BUFFER`

		.. warning::

		   Setting to this attribute binds the VAO containing it.

		:param value: The data for the attribute.
		:type value: :py:class:`.BufferItem`
		'''
		return self._data

	@data.setter
	def data(self, value):
		if value.components < self.components:
			# Never allow, GL_ARB_vertex_attrib_64bit suggests default values are for compatability
			raise ValueError("Specified only {} components for a vertex attribute expecting {}."
			                 .format(value.components, self.components))
		with self.vao, value.buffer.bind(GL.GL_ARRAY_BUFFER):
			setter = self.gl_pointer_functions[self.datatype.scalar_type]
			for location in self.locations:
				setter(location, self.components, numpy_buffer_types[value.dtype.base],
					   self.normalized, value.buffer.stride, GL.GLvoidp(value.offset))
		self._data = value

	@property
	def divisor(self):
		return self._divisor

	@divisor.setter
	def divisor(self, value):
		'''The divisor for this vertex attribute. If it is 0, one element of the attribute will be
		consumed per vertex rendered. If it is non-zero, one element will be consumed for each
		``divisor`` instances of the shader invoked.

		:param int value: The number of instances to render using each value of this attribute, or
		  ``0`` to use one value per vertex.
		'''
		with self.vao:
			GL.glVertexAttribDivisor(self.location, value)
		self._divisor = value
