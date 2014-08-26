from OpenGL import GL

from itertools import chain, repeat, product as cartesian
from collections import Counter, namedtuple

from .datatypes import GLSLVar, GLSLType
from .datatypes import gl_types, gl_integer_types, data_types, prefixes, vector_sizes
from .buffers import Buffer

from util.misc import equal, product, subIter

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

	:param [GLSLVar] attributes: The attributes that the VAO will contain.
		Locations will be automatically generated.
	:param handle: The OpenGL handle to use. One will be created if it is None
	:type param: int or None'''

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
		return self.element_buffer
		raise NotImplementedError("Element read access not implemented.")
	
	@elements.setter
	def elements(self, value):
		'''Set an ElementBuffer to be used with this VAO'''

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
		if not self.bound:
			GL.glBindVertexArray(self.handle)
		self.bound += 1
	
	def __exit__(self, ex, val, tr):
		self.bound -= 1
		if not self.bound:
			GL.glBindVertexArray(0)

class VertexAttribute(GLSLVar):
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
		return gl_type_indices[self.type]
	
	@property
	def indices(self):
		return self.type_indices * product(self.shape)
	
	@property
	def data(self):
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
VertexFormat = namedtuple("VertexFormat", ["dtype", "offset", "components", "stride"])

class VertexDataTrack:
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
		self.normalize = normalize
		self.indices = indices

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
	target = GL.GL_ARRAY_BUFFER

	def __init__(self, *contents, usage=GL.GL_DYNAMIC_DRAW, handle=None):
		super().__init__(usage, handle)
		self.blocks = []
		for c in contents:
			try:
				self.blocks.append(VertexDataBlock(self, *c))
			except TypeError:
				self.blocks.append(VertexDataBlock(self, c))

	def __len__(self):
		return min(len(c) for c in self.blocks)

	def __getitem__(self, i):
		raise NotImplementedError("Vertex buffer access is not yet implemented.")

	def __setitem__(self, i, values):
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
	target = GL.GL_ELEMENT_ARRAY_BUFFER
	def __init__(self, usage=GL.GL_DYNAMIC_DRAW, handle=None):
		super().__init__(usage, handle)
		self.dtype = None
		self.length = 0
	
	def __len__(self):
		return self.length

	def __setitem__(self, i, values):
		if gl_types[values.dtype] not in gl_integer_types:
			raise ValueError("Array dtype '{}' is not a valid dtype for an element buffer".format(values.dtype))
		values.shape = (-1,)
		if i is not Ellipsis:
			i = slice(*(s * self.dtype.itemsize for s in i.indices()))
		self.bytes[i] = values
		self.length = len(values)
		self.dtype = values.dtype
