from OpenGL import GL

from itertools import chain, repeat, product as cartesian
from collections import Counter, namedtuple

from .datatypes import GLSLVar, GLSLType
from .datatypes import gl_types, gl_integer_types, data_types, prefixes, vector_sizes
from .buffers import Buffer

from util.misc import product, subIter

import numpy

from ctypes import c_void_p

gl_type_indices = { data_type: 1 for data_type in data_types }
gl_type_indices.update({ "{}vec{}".format(prefix, size): 1 for prefix, size
					  in cartesian(prefixes.values(), vector_sizes) })
gl_type_indices.update({ "mat{}".format(size): size for size in vector_sizes })
gl_type_indices.update({ "mat{}x{}".format(size1, size2): size1 for size1, size2
					  in cartesian(vector_sizes, repeat=2) })

# TODO: Allow specifying attribute locations in advance
class VAO:
	'''A class to represent VAO objects.

	:param attributes: The attributes that the VAO will contain.
		Locations will be automatically generated.
	:type attributes: [:py:class:`.GLSLVar`]
	:param handle: The OpenGL handle to use. One will be created if it is :py:obj:`None`
	:type param: :py:obj`int` or :py:obj:`None`
	'''

	def __init__(self, *attributes, handle=None):
		self.handle = GL.glGenVertexArrays(1) if handle is None else handle
		self.bound = 0
		self.element_buffer = None

		self.attributes = []
		index = 0
		for a in attributes:
			vertex_attribute = VertexAttribute.fromGLSLVar(index, self, a)
			self.attributes.append(vertex_attribute)
			index += vertex_attribute.indices

		with self:
			for attribute in self.attributes:
				GL.glEnableVertexAttribArray(attribute.location)
	
	@property
	def elements(self):
		'''The element buffer to be used with this VAO for indexed drawing.

		.. admonition:: |buffer-bind|

		   Setting to this property binds the buffer being assigned to the VAO

		.. admonition:: |vao-bind|

		   Setting to this property binds the VAO being assigned to
		'''
		return self.element_buffer
	
	@elements.setter
	def elements(self, value):
		if value.target != GL.GL_ELEMENT_ARRAY_BUFFER:
			raise ValueError("Buffer target is not GL_ELEMENT_ARRAY_BUFFER")
		with self:
			GL.glBindBuffer(value.target, value.handle)
		GL.glBindBuffer(value.target, 0)
		self.element_buffer = value
	
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

class VertexAttribute(GLSLVar):
	"""A vertex attribute. Should not be instantiated directly, but instead be accessed through its
	parent VAO.
	"""

	def __init__(self, location, vao, name, gl_type, shape=1):
		super().__init__(name=name, gl_type=gl_type, shape=shape)
		self.location = location
		self.vao = vao
		self.track = None
	
	@classmethod
	def fromGLSLVar(cls, location, vao, var):
		return cls(location, vao, var.name, var.type, var.shape)

	def __str__(self):
		base = GLSLVar.__str__(self)
		if self.location is not None:
			layout = "layout(location={})".format(self.location)
			return ' '.join((layout, 'in', base))
		return base

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
	def data(self):
		'''The data track in a buffer backing this vertex attribute.

		.. admonition:: |buffer-bind|

		   Setting to this property binds the buffer that contains the track being assigned

		.. admonition:: |vao-bind|

		   Setting to this property binds the VAO that contains the attribute being assigned to
		'''
		raise NotImplementedError("Vertex Attribute data access not implemented.")
	
	@data.setter
	def data(self, track):
		if track.indices < self.indices:
			raise ValueError("Data track does not fill all vertex attribute indices.")

		with self.vao, track.block.buf:
			for i in range(self.indices):
				GL.glVertexAttribPointer( self.location + i, track.components, gl_types[track.dtype.base]
										, track.normalize, track.stride, c_void_p(track.offset))
		
		if self.track is not None:
			self.track.pointers.remove(self)
		self.track = track
		self.track.pointers.add(self)

SubBuffer = namedtuple("SubBuffer", ["offset", "nbytes"])
# CONTINUE HERE: Get rid of this? Perhaps update tracks directly and then add a refresh method?
VertexFormat = namedtuple("VertexFormat", ["dtype", "offset", "components", "stride"])

class VertexDataTrack:
	'''A single interleaved track in a VertexDataBlock. This class keeps track of which
	VertexAttributes reference it, and update them whenever its data format is changed. Should not
	be instantiated directly, but instead referenced through its parent block.

	:param block: The block containing this track
	:type block: :py:class:`.VertexDataBlock`
	:param dtype: The dtype of the data in this track
	:type dtype: :py:class:`numpy.dtype` or :py:obj:`None`
	:param int components: The number of components specified in this data track.
	:param int indices: The number of vertex attribute indices specified in this data track.
	:param bool normalize: Whether to normalize integer data into the [0,1] range
	'''
	gl_type_components = {data_type: 1 for data_type in data_types}
	gl_type_components.update({ "{}vec{}".format(prefix, size): size for prefix, size
	                         in cartesian(prefixes.values(), vector_sizes) })
	gl_type_components.update({ "mat{}".format(size): size ** 2
	                         for size in vector_sizes })
	gl_type_components.update({ "mat{}x{}".format(size1, size2): size1 * size2
	                         for size1, size2 in cartesian(vector_sizes, repeat=2) })

	def __init__(self, block, offset, dtype, components, indices, normalize=True):
		self.block = block
		self.offset = offset
		self.dtype = dtype
		self.components = components
		self.indices = indices
		self.normalize = normalize

		self.pointers = set()
	
	@classmethod
	def fromGLSLType(cls, block, var):
		return cls(block, None, None
		          , cls.gl_type_components[var.type]
		          , gl_type_indices[var.type] * var.count)

	@property
	def stride(self):
		return self.block.stride
	
	def setFormat(self, fmt):
		self.dtype = fmt.dtype
		self.offset = fmt.offset
		self.components = fmt.components
		for ptr in self.pointers:
			ptr.data = self
	
	def __repr__(self):
		return "{}(block={}, offset={})".format(type(self).__name, self.block, self.offset)
	
class VertexDataBlock:
	'''This class describes a contiguous block of a buffer, consisting of multiple interleaved
	:py:class:`.VertexDataTrack`. Should not be instantiated directly, but instead accessed through
	its parent buffer.

	:param buf: The buffer this block belongs to.
	:type buf: :py:class:`.VertexBuffer`
	:param contents: The vertex attributes this block describes.  One track will be created for each
		attribute.
	:type contents: :py:class:`.GLSLType`
	:param location: The location of this block in the buffer passed in ``buf``
	'''
	def __init__(self, buf, *contents, location=(None, None)):
		self.buf = buf
		self.location = SubBuffer(*location)
		self.tracks = [VertexDataTrack.fromGLSLType(self, c) for c in contents]

		self.length = 0
		self.stride = None
		self.dtype = None

	def __len__(self):
		return self.length

	def __getitem__(self, i):
		raise NotImplementedError("Vertex Buffer access is not yet implemented.")

	def cast(self, dtype):
		'''Generate a new dtype based on the block's contents and a passed dtype.

		:param numpy.dtype dtype: the dtype to use as a base
		:raises ValueError: if ``dtype`` is a simple dtype and not a valid OpenGL type
		:raises ValueError: if ``dtype`` is a record dtype containing types that are not valid
			OpenGL types
		:raises ValueError: if ``dtype`` is a record dtype with a length not equal to the number of
			tracks in the block
		'''

		if len(dtype) == 0:
			if dtype.base.base not in gl_types:
				raise ValueError("Invalid data type for a vertex attribute.")
			new_dtype = [('', dtype.base.base, (c.indices, c.components)) for c in self.tracks]
		else:
			if len(dtype) != len(self.tracks):
				raise ValueError("dtype length does not match number of data tracks.")
			if any(len(dt) > 1 for dt in subIter(dtype)):
				raise ValueError("Vertex attribute data may not be a record dtype.")
			dtype = [dt if len(dt) == 0 else dt[0] for dt in subIter(dtype)]
			if any(dt.base.base not in gl_types for dt in subIter(dtype)):
				raise ValueError("Invalid vertex attribute data type in dtype.")
			new_dtype = [('', dt.base.base, (c.indices, c.components))
			             for dt, c in zip(subIter(dtype), self.tracks)]
		return numpy.dtype(new_dtype)

	def __setitem__(self, i, value):
		'''Set vertices described by the data block. All attributes contained in the block must be
		specified.

		:param i: The section of the block to be set, in vertex indices
		:type i: :py:obj:`slice` or :py:obj:`int` or :py:obj:`Ellipsis`
		:param numpy.ndarray value: The new data for the block
		:raises RuntimeError: if the block is set to before its location is defined (e.g. by setting
			the contents of the entire VertexBuffer
		:raises IndexError: if ``i`` does not represent a contiguous section of the buffer.
		:raises ValueError: if the length of ``i`` does not match the length of ``value``
		:raises ValueError: if a section of the buffer is being set to a different dtype than the
			rest

		.. admonition |buffer-bind|

		   Setting to a data block binds the buffer it belongs to.

		.. admonition |vao-bind|

		   Setting to a data block binds all of the VAOs that reference its contents.
		'''
		if None in self.location:
			raise RuntimeError("Cannot set a buffer block before it's location has been defined.")

		value = numpy.asarray(value)
		new_dtype = self.cast(value.dtype)
		if len(value.dtype) == 0:
			if not value.dtype.shape:
				value.dtype = new_dtype
				value = value.ravel()
			else:
				raise NotImplementedError("Setting complex non-record dtypes is not yet implemented.")
		else:
			value.dtype = new_dtype
			value = value.ravel()

		if i is Ellipsis:
			i = slice(0, len(value))
		else:
			if not isinstance(i, slice):
				i = slice(i, i + 1)
			else:
				i = slice(*i.indices(len(self)))
			if i.step != 1:
				raise IndexError("Only contiguous buffer sections may be set.")
			if len(value) != i.stop - i.start:
				raise ValueError("New data is a different length to data being set.")
			if value.dtype != self.dtype:
				raise ValueError("Value dtype does not match rest of buffer.")

		start = self.location.offset + value.dtype.itemsize * i.start
		end = self.location.offset + value.dtype.itemsize * i.stop
		if end - start > self.location.nbytes:
			raise IndexError("Cannot write outside of buffer block bounds.")

		if i is Ellipsis:
			self.dtype = value.dtype
			self.length = len(value)

		self.stride = value.itemsize
		with self.buf:
			self.buf.bytes[start:end] = value
			for (dt, offset), track in zip(sorted(value.dtype.fields.values(), key=lambda x: x[1]), self.tracks):
				track.setFormat(VertexFormat(dt.base, offset + self.location.offset, dt.shape[-1], self.stride))

class VertexBuffer(Buffer):
	'''A buffer that contains vertex data. A buffer contains one or more VertexDataBlocks, each of
	which contains one or more VertexDataTracks of interleaved vertex data.

	:param contents: The vertex attributes to be stored in this buffer, grouped into blocks. Single
		attributes will be assigned to their own block.
	:type contents: :py:class:`.GLSLType` or [:py:class:`.GLSLType`]
	'''

	target = GL.GL_ARRAY_BUFFER
	'''This buffer binds to ``GL.GL_ARRAY_BUFFER``'''

	def __init__(self, *contents, usage=GL.GL_DYNAMIC_DRAW, handle=None):
		super().__init__(usage, handle)
		self.blocks = []
		for c in contents:
			try:
				self.blocks.append(VertexDataBlock(self, *c))
			except TypeError:
				self.blocks.append(VertexDataBlock(self, c))

	def __len__(self):
		'''Returns the number of vertices described in the smallest data block
		contained in this buffer
		'''
		return min(len(c) for c in self.blocks)

	def __getitem__(self, i):
		raise NotImplementedError("Vertex buffer access is not yet implemented.")

	def __setitem__(self, i, values):
		'''Set the contents of this vertex buffer. Currently, only setting the
		whole buffer (``i == Ellipsis``) is implemented.

		:param Ellipsis i: The section of the buffer to be set. If Ellipsis
			(``...``) is passed, the buffer will be resized to fit the data.
		:param value: The new data for the buffer. Must be an
			iterable of the same length as the number of blocks being set.
		:type value: [:py:class:`numpy.ndarray`]

		.. admonition:: |buffer-bind|

		   This method binds the buffer being set to.
		'''
		if i is Ellipsis:
			tmp = numpy.empty(sum(v.nbytes for v in values), dtype='bytes')

			# Setting dtype in for loop doesn't stick on next loop
			new_dtypes = [block.cast(v.dtype) for block, v in zip(self.blocks, values)]
			offset = 0
			for block, value, new_dtype in zip(self.blocks, values, new_dtypes):
				value.dtype = new_dtype
				value = value.ravel()
				block.length = len(value)
				block.dtype = value.dtype
				block.location = SubBuffer(offset, value.nbytes)
				block.stride = value.dtype.itemsize

				value.dtype = tmp.dtype
				tmp[offset:offset + value.nbytes] = value
				offset += value.nbytes

			with self:
				self.bytes[i] = tmp
				for block, value, dtype in zip(self.blocks, values, new_dtypes):
					for (dt, offset), track in zip(sorted(dtype.fields.values(), key=lambda x: x[1]), block.tracks):
						track.setFormat(VertexFormat(dt, offset + block.location.offset, dt.shape[-1], block.stride))
		else:
			raise NotImplementedError("Setting whole buffer subsets is not yet implemented.")
	
class ElementBuffer(Buffer):
	'''An element array buffer for indexed drawing.'''

	target = GL.GL_ELEMENT_ARRAY_BUFFER
	'''This buffer binds to ``GL.GL_ELEMENT_ARRAY_BUFFER``'''
	def __init__(self, usage=GL.GL_DYNAMIC_DRAW, handle=None):
		super().__init__(usage, handle)
		self.dtype = None
		self.length = 0
	
	def __len__(self):
		return self.length

	def __setitem__(self, i, values):
		'''Set the contents of the array buffer.

		:param i: The buffer slice to be set, if it is :py:obj:`Ellipsis`, the buffer will be
			resized to fit the data.
		:type i: :py:obj:`slice` or :py:obj:`int` or :py:obj:`Ellipsis`
		:param numpy.ndarray values: The data for the buffer.
		:raises ValueError: If the new data is not an integer type compatible with OpenGL

		.. admonition:: |buffer-bind|

		   This method binds the buffer it belongs to.
		'''
		if gl_types[values.dtype] not in gl_integer_types:
			raise ValueError("Array dtype '{}' is not a valid dtype for an element buffer".format(values.dtype))
		values.shape = (-1,)
		if i is not Ellipsis:
			i = i if isinstance(i, slice) else slice(i, i + 1)
			i = range(*i.indices(len(self)))
			i = range(i.start * self.dtype.itemsize, i.stop * self.dtype.itemsize)
		self.bytes[i] = values
		self.length = len(values)
		self.dtype = values.dtype
