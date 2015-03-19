from OpenGL import GL
from numpy import ndarray, dtype

from itertools import chain, repeat
from contextlib import contextmanager

from ctypes import c_byte

from util.misc import product, contains
from util.indexing import isContiguous, flatOffset

# Note: GL_INT_2_10_10_10_REV, GL_UNSIGNED_INT_2_10_10_10_REV, GL_UNSIGNED_INT_10F_11F_11F_REV are not possible using numpy dtypes.
numpy_buffer_types = { dtype('int8'): GL.GL_BYTE
                     , dtype('uint8'): GL.GL_UNSIGNED_BYTE
                     , dtype('int16'): GL.GL_SHORT
                     , dtype('uint16'): GL.GL_UNSIGNED_SHORT
                     , dtype('int32'): GL.GL_INT
                     , dtype('uint32'): GL.GL_UNSIGNED_INT
                     , dtype('float16'): GL.GL_HALF_FLOAT
                     , dtype('float32'): GL.GL_FLOAT
                     , dtype('float64'): GL.GL_DOUBLE }
buffer_numpy_types = {v: k for k, v in numpy_buffer_types.items()}
integer_buffer_types = { GL.GL_BYTE, GL.GL_UNSIGNED_BYTE
                       , GL.GL_SHORT, GL.GL_UNSIGNED_SHORT
                       , GL.GL_INT, GL.GL_UNSIGNED_INT }
floating_point_buffer_types = { GL.GL_HALF_FLOAT, GL.GL_FLOAT, GL.GL_DOUBLE }

def baseDtype(dtype):
	dt, shape = dtype.subdtype
	while dt.subdtype:
		dt, new_shape = dt.subdtype
		shape += new_shape
	return dt, shape

class Buffer:
	"""An OpenGL buffer.

	A buffer is defined by the data it contains. The data type of a buffer object is always a record
	data type. Individual blocks can be accessed through the :py:class:`.SubBuffer` objects obtained
	by indexing.

	:keyword usage: The intended usage of the buffer.
	:keyword handle: The buffer handle. One will be created if it is not provided.
	:type handle: :py:obj:`int` or :py:obj:`None`
	"""

	def __init__(self, data=None, usage=GL.GL_DYNAMIC_DRAW, handle=None):
		self.mapped = False
		self.usage = usage
		self.dtype = None
		# Buffer not created until bound, this only reserves a name
		self.handle = GL.glGenBuffers(1) if handle is None else handle
		if not self.handle:
			raise RuntimeError("Failed to generate buffer.")
		self.active_bindings = set()

	# TODO: Deal with deleting buffers
	def map(self, access=(GL.GL_MAP_READ_BIT | GL.GL_MAP_WRITE_BIT)):
		"""Returns a numpy array mapped to the entire buffer.

		:param access: The OpenGL flags passed to map buffer range.

		.. warning::

		   |buffer-bind|
		"""

		if self.mapped:
			raise RuntimeError("Buffer is already mapped.")
		try:
			binding = next(iter(self.active_bindings))
		except StopIteration:
			raise RuntimeError("Buffers can only be mapped if they are bound to a target.")
		mem = GL.glMapBufferRange(binding, 0, self.dtype.itemsize, access)
		self.mapped = True

		numpy_buffer = (c_byte * self.dtype.itemsize).from_address(mem)
		return BufferMapping(self.dtype.shape, self.dtype.base, numpy_buffer, self, 0, access)

	def unmap(self):
		"""Unmaps the buffer. Any mapped arrays *must not* be used after this action. This implicitly
		flushes any changes made to the buffer.

		.. warning::

		   |buffer-bind|
		"""

		if not self.mapped:
			return
		try:
			binding = next(iter(self.active_bindings))
		except StopIteration:
			raise RuntimeError("Buffers can only be unmapped if they are bound to a target.")
		GL.glUnmapBuffer(binding)
		self.mapped = False

	def __setitem__(self, idxs, data):
		"""Set the contents of the buffer. Currently, only :py:obj:`Ellipsis` is valid as an index.

		:param idxs: The indices to set. A new buffer will be created if :py:obj:`Ellipsis` is
			passed.
		:param data: The new contents of the buffer. If a :py:class:`numpy.ndarray` is passed, it
			will be used to initialize the buffer. If a :py:class:`numpy.dtype`
			is passed, the buffer will not be initialized.
		:type data: :py:class:`numpy.dtype` or :py:class:`numpy.ndarray`

		.. warning::

		   |buffer-bind|
		"""
		if idxs is Ellipsis:
			if isinstance(data, dtype):
				dt = data
				data = None
			else:
				dt = ( data.dtype if product(data.shape) == 1
				       else dtype((data.dtype, data.shape)) )
			try:
				binding = next(iter(self.active_bindings))
			except StopIteration:
				raise RuntimeError("Buffer data can only be set if it is bound.")
			GL.glBufferData(binding, dt.itemsize, data, self.usage)
			self.dtype = dt
		else:
			raise NotImplementedError("TODO: Allow changing replacing buffer dtypes.")

	def __getitem__(self, idxs):
		if not isinstance(idxs, tuple):
			idxs = (idxs,)
		return SubBuffer(self, *idxs)

	@contextmanager
	def bind(self, target, index=None):
		"""Binds the buffer to a target, with an optional index for an indexed target.

		.. warning::

		   It is not allowed to bind two buffers (or one buffer twice) to the same target
		   simultaneously.
		"""

		if index is None:
			GL.glBindBuffer(target, self.handle)
		else:
			GL.glBindBufferBase(target, index, self.handle)
		self.active_bindings.add(target)
		yield
		if index is None:
			GL.glBindBuffer(target, 0)
		else:
			GL.glBindBufferBase(target, index, 0)
		self.active_bindings.remove(target)

	@property
	def nbytes(self):
		return self.dtype.itemsize

	@property
	def stride(self):
		"""Return the stride of buffer rows (in machine units)."""
		try:
			count, *shape = self.dtype.shape
		except ValueError:
			return None
		return self.dtype.base.itemsize * product(shape)

	@property
	def items(self):
		'''Returns the items in the buffer, suitable for passing to a :py:class:`VAOAttribute`'''

		return BufferItem.fromBuffer(self)

	@property
	def data(self):
		'''Returns or sets the contents of the buffer, as a :py:class:`numpy.ndarray`

		.. warning::

		   |buffer-bind|
		'''
		# TODO: server-side buffer copies via assigning a (Sub)Buffer (instead of numpy array)
		try:
			binding = next(iter(self.active_bindings))
		except StopIteration:
			raise RuntimeError("Buffer contents can only be retrieved if they are bound.")
		a = GL.glGetBufferSubData(binding, 0, self.nbytes)
		base_dtype, shape = baseDtype(self.dtype)
		a.dtype = base_dtype
		a.shape = shape
		return a

	@data.setter
	def data(self, value):
		try:
			binding = next(iter(self.active_bindings))
		except StopIteration:
			raise RuntimeError("Buffer contents can only be set if they are bound.")
		value.dtype = self.dtype.base
		value.shape = self.dtype.shape
		GL.glBufferSubData(binding, 0, value.nbytes, value)

# FIXME: This might be able to inherit from buffer? Or vice versa?
class SubBuffer:
	'''A *contiguous* section of a buffer. All values are calculated on initialization, so if the
	underlying buffer changes it's dtype the view must not be used.

	:param parent: The parent containing this block.
	:type parent: :py:class:`Buffer` or :py:class:`SubBuffer`
	:param \\*idxs: The indices into the parent that define the sub-buffer.
	:type \\*idxs: :py:obj:`int` or :py:obj:`slice` or :py:obj:`str` or [:py:obj:`str`]

	:raises IndexError: If the specified indices do not represent a contiguous section of the
		parent.
	'''
	
	# FIXME: Reduce buffer indexing options to simplify this.
	def __init__(self, parent, *idxs):
		self.parent = parent
		if parent.dtype.subdtype is None:
			# Parent is a record array
			# Only allow one field for consistency with numpy.
			if len(idxs) != 1:
				raise IndexError("Invalid index into record array.")
			idx = idxs[0]
			if idx in range(-len(parent.dtype), len(parent.dtype)):
				idx = idx % len(parent.dtype)
			else:
				try:
					idx = parent.dtype.names.index(idx)
				except ValueError:
					raise IndexError("No such field: {}".format(idx))
			offset = sum(parent.dtype[i].itemsize for i in range(idx))
			self.dtype = parent.dtype.base[idx]
		else:
			parent_base, parent_shape = parent.dtype.subdtype
			if all(isinstance(idx, slice) or isinstance(idx, int) for idx in idxs):
				# Indexing shape
				if len(idxs) > len(parent_shape):
					raise IndexError("Too many indices.")
				if not isContiguous(idxs, parent_shape):
					raise IndexError("Non-contiguous indexing is not permitted.")
				offset = flatOffset(idxs, parent_shape, base=parent_base.itemsize)
				if any(idx >= s for idx, s in zip(idxs, parent_shape)
				       if not isinstance(idx, slice)):
					raise IndexError("Index out of bounds.")
				idxs = chain(idxs, repeat(slice(None)))
				shape = (len(range(*idx.indices(s))) for idx, s in zip(idxs, parent_shape)
				         if isinstance(idx, slice))
				self.dtype = dtype((parent_base, tuple(shape)))
			else:
				# Indexing record fields
				# Multiple indices must be in a list for consistency with numpy
				if len(idxs) > 1:
					raise IndexError("Invalid indexes.")
				if isinstance(idxs[0], str):
					field_name = first_field = idxs[0]
					if field_name not in parent_base.names:
						raise IndexError("No such field: {}".format(field_name))
					if len(parent_base) > 1 and product(parent_shape) > 1:
						raise IndexError("Non-contiguous indexing is not permitted.")
					field = parent_base[field_name]
					self.dtype = dtype((field.base, parent_shape + field.shape))
				else:
					field_names = tuple(filter(lambda x: x in parent_base.names, idxs[0]))
					try:
						first_field = field_names[0]
					except IndexError:
						raise IndexError("No such fields: {}".format(', '.join(field_names)))
					if product(parent_shape) > 1:
						if field_names != parent_base.names:
							raise IndexError("Non-contiguous indexing is not permitted.")
					else:
						if not contains(field_names, parent_base.names):
							raise IndexError("Non-contiguous indexing is not permitted.")
					base_dtype = dtype([(name, parent_base[name]) for name in field_names])
					self.dtype = dtype((base_dtype, parent_shape))
				offset = parent_base.fields[first_field][1]
		self.offset = offset + getattr(parent, 'offset', 0)
		self.buffer = getattr(parent, 'buffer', parent)

	@property
	def items(self):
		'''Returns the items in the buffer, suitable for passing to a :py:class:`VAOAttribute`'''
		return BufferItem.fromBuffer(self)

	def map(self, access=(GL.GL_MAP_READ_BIT | GL.GL_MAP_WRITE_BIT )):
		"""Returns a numpy array mapped to the sub-buffer.

		:param access: The OpenGL flags passed to map buffer range.

		.. warning::

		   |buffer-bind|
		"""
		try:
			binding = next(iter(self.buffer.active_bindings))
		except StopIteration:
			raise RuntimeError("Sub-buffer contents can only be set if the buffer is bound.")
		mem = GL.glMapBufferRange(binding, self.offset, self.dtype.itemsize, access)
		self.buffer.mapped = True

		numpy_buffer = (c_byte * self.dtype.itemsize).from_address(mem)
		return BufferMapping(self.dtype.shape, self.dtype.base, numpy_buffer,
		                     self.buffer, self.offset, access)

	def __getitem__(self, idxs):
		if not isinstance(idxs, tuple):
			idxs = (idxs,)
		return SubBuffer(self, *idxs)

	def __setitem__(self, idxs, value):
		raise NotImplementedError("TODO: Allow changing sub-buffer dtypes.")

	@property
	def nbytes(self):
		return self.dtype.itemsize

	@property
	def data(self):
		"""See :py:meth:`Buffer.data`

		.. warning::

		   |buffer-bind|
		"""
		try:
			binding = next(iter(self.buffer.active_bindings))
		except StopIteration:
			raise RuntimeError("Sub-buffer contents can only be retrieved if the buffer is bound.")
		a = GL.glGetBufferSubData(binding, self.offset, self.nbytes)
		a.dtype = self.dtype.base
		a.shape = self.dtype.shape
		return a

	@data.setter
	def data(self, value):
		try:
			binding = next(iter(self.buffer.active_bindings))
		except StopIteration:
			raise RuntimeError("Sub-buffer contents can only be set if the buffer is bound.")
		value.dtype = self.dtype.base
		value.shape = self.dtype.shape
		GL.glBufferSubData(binding, self.offset, value.nbytes, value)

	@property
	def stride(self):
		"""Return the stride of buffer rows (in machine units)."""
		try:
			count, *shape = self.dtype.shape
		except ValueError:
			return None
		return self.dtype.base.itemsize * product(shape)

class BufferItem:
	"""A repeated item in a buffer. Intended for use with vertex attributes.

	:ivar dtype: The data type of this item
	:ivar buffer: The buffer this item is from
	:ivar offset: The offset of this item in the buffer
	"""

	def __init__(self, parent, offset, dtype):
		self.dtype = dtype
		self.offset = offset + getattr(parent, 'offset', 0)
		self.buffer = getattr(parent, 'buffer', parent)

	@classmethod
	def fromBuffer(cls, buf):
		base, shape = buf.dtype.subdtype or (buf.dtype, ())
		dt = dtype((base, shape[1:]))
		return cls(buf, 0, dt)

	@property
	def components(self):
		return product(self.dtype.shape)

	def __getitem__(self, idx):
		if len(self.dtype):
			# Record data type
			try:
				dt, offset = self.dtype.fields[idx]
			except KeyError:
				dt, offset = self.dtype.fields[self.dtype.names[idx]]
		else:
			# Array data type
			try:
				count, *shape = self.dtype.shape
			except ValueError:
				raise IndexError("Cannot index into a base data type.")
			if idx >= count:
				raise IndexError("Index {} is out of bounds for array of length {}"
				                 .format(idx, count))
			dt = dtype((self.dtype.base, tuple(shape)))
			offset = self.dtype.itemsize * idx

		return BufferItem(self, offset, dt)

class BufferMapping(ndarray):
	"""A numpy array that is mapped to a buffer.

	Care must be taken not to access this array after it has been un-mapped.

	.. warning:: Undefined behaviour

	   Behavious is undefined if the array is used in a manner inconsistent with the flags it
	   was mapped with (e.g. reading an array mapped only with :py:obj:`GL.GL_MAP_WRITE_BIT`)
	"""
	def __new__(cls, shape, dtype, buffer, gl_buffer, gl_offset, access):
		obj = ndarray.__new__(cls, shape, dtype, buffer)
		obj.gl_buffer = gl_buffer
		obj.offset = gl_offset
		obj.access = access
		return obj

	def __array_finalize__(self, obj):
		if obj is None:
			# in __new__, will set properties there
			return
		elif isinstance(obj, BufferMapping):
			# new-from-template (indexing), copy over properties
			self.gl_buffer = obj.gl_buffer
			self.offset = obj.offset
			self.access = obj.access
		else:
			# View casting, not allowed since we need to use a GL memory location.
			raise TypeError("Cannot cast to a mapped buffer.")

	def flush(self):
		"""Ensures changes are visible to GL for rendering. This has two actions depending on how
		the buffer was mapped.

		If it was mapped with :py:obj:`GL.GL_MAP_FLUSH_EXPLICIT_BIT` the buffer is flushed using
		:py:func:`GL.glFlushMappedBufferRange`

		Otherwise, a memory barrier is issued. This requires ``ARB_shader_image_load_store``.

		.. warning::

		   |buffer-bind|
		"""

		# FIXME: FLUSH_EXPLICIT_BIT is for flushing sub-ranges (no auto-flush at unmap?), so flush minimal region instead of whole mapped region
		try:
			binding = next(iter(self.gl_buffer.active_bindings))
		except StopIteration:
			raise RuntimeError("Buffer mappings can only be flushed if the buffer is bound.")

		if self.access & GL.GL_MAP_FLUSH_EXPLICIT_BIT:
			length = self.dtype.itemsize * product(self.shape)
			GL.glFlushMappedBufferRange(binding, self.offset, length)
		else:
			GL.glMemoryBarrier(GL.GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT)
