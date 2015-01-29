from OpenGL import GL
from OpenGL.constants import GLboolean, GLint, GLuint, GLfloat, GLdouble
from numpy import ndarray, dtype, frombuffer

from itertools import chain, repeat, takewhile

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
		self.bound = 0
		self.mapped = False
		self.usage = usage
		self.dtype = None
		self.handle = GL.glGenBuffers(1) if handle is None else handle

	# TODO: Deal with deleting buffers

	def map(self, access=(GL.GL_MAP_READ_BIT | GL.GL_MAP_WRITE_BIT)):
		"""Returns a numpy array mapped to the entire buffer.

		:param access: The OpenGL flags passed to map buffer range.

		.. admonition:: |buffer-bind|

		   This action binds the buffer.
		"""

		with self:
			mem = GL.glMapBufferRange(self.target, 0, self.dtype.itemsize, access)
		self.mapped = True

		numpy_buffer = (c_byte * self.dtype.itemsize).from_address(mem)
		return BufferMapping(self.dtype.shape, self.dtype.base, numpy_buffer, self, 0, access)

	def unmap(self):
		"""Unmaps the buffer. Any mapped arrays *must not* be used after this action. This implicitly
		flushes any changes made to the buffer.

		.. admonition:: |buffer-bind|

		   This action binds the buffer.
		"""
		if not self.mapped:
			return

		with self:
			GL.glUnmapBuffer(self.target)
		self.mapped = False

	def __setitem__(self, idxs, data):
		"""Set the contents of the buffer. Currently, only :py:obj:`Ellipsis` is valid as an index.

		:param idxs: The indices to set. A new buffer will be created if :py:obj:`Ellipsis` is
			passed.
		:param data: The new contents of the buffer. If a :py:class:`numpy.ndarray` is passed, it
			will be used to initialize the buffer. If a :py:class:`numpy.dtype`
			is passed, the buffer will not be initialized.
		:type data: :py:class:`numpy.dtype` or :py:class:`numpy.ndarray`

		.. admonition:: |buffer-bind|

		   This action binds the buffer.
		"""
		if idxs is Ellipsis:
			if isinstance(data, dtype):
				dt = data
				data = None
			else:
				dt = ( data.dtype if product(data.shape) == 1
				       else dtype((data.dtype, data.shape)) )
			with self:
				GL.glBufferData(self.target, dt.itemsize, data, self.usage)
			self.dtype = dt

		else:
			raise NotImplementedError("TODO: Allow changing replacing buffer dtypes.")

	def __getitem__(self, idxs):
		if not isinstance(idxs, tuple):
			idxs = (idxs,)
		return SubBuffer(self, *idxs)

	# FIXME: create multiple context managers for different binding targets.
	def __enter__(self):
		"""Buffer objects provide a context manager. This keeps track of how
		many times the buffer has been bound and unbound. Grouping operations
		on a buffer within a context where it is bound prevents repeated
		binding and un-binding.

		.. _buffer-bind-warning:
		.. warning::
		   It is not allowed to bind two buffers with the same target
		   simultaneously. It is allowed to bind the *same* buffer multiple times.
		   
		   Methods that bind a buffer will be documented"""
		if not self.bound:
			GL.glBindBuffer(self.target, self.handle)
		self.bound += 1

	def __exit__(self, ex, val, tr):
		self.bound -= 1
		if not self.bound:
			GL.glBindBuffer(self.target, 0)

	@property
	def nbytes(self):
		return self.dtype.itemsize

	@property
	def data(self):
		""" Returns or sets the contents of the buffer, as a :py:class:`numpy.ndarray`

		.. admonition:: |buffer-bind|

		   Reading and writing from this property both bind the buffer.
		"""
		with self:
			a = GL.glGetBufferSubData(self.target, 0, self.nbytes)
		a.dtype = self.dtype.base
		a.shape = self.dtype.shape
		return a

	@data.setter
	def data(self, value):
		value.dtype = self.dtype.base
		value.shape = self.dtype.shape
		with self:
			GL.glBufferSubData(self.target, 0, value.nbytes, value)

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
	
	def __init__(self, parent, *idxs):
		self.parent = parent
		if parent.dtype.subdtype is None:
			# Parent is a record array
			# Only allow one field for consistency with numpy.
			if len(idxs) != 1:
				raise IndexError("Invalid index into record array.")
			idx = idxs[0]
			try:
				idx = parent.dtype.names.index(idx)
			except ValueError:
				if idx not in range(-len(parent.dtype), len(parent.dtype)):
					raise IndexError("No such field: {}".format(idx))
			self.offset = sum(parent.dtype[i].itemsize for i in range(idx))
			self.dtype = parent.dtype.base[idx]
		else:
			parent_base, parent_shape = parent.dtype.subdtype
			if all(isinstance(idx, slice) or isinstance(idx, int) for idx in idxs):
				# Indexing shape
				if len(idxs) > len(parent_shape):
					raise IndexError("Too many indices.")
				if not isContiguous(idxs, parent_shape):
					raise IndexError("Non-contiguous indexing is not permitted.")
				self.offset = flatOffset(idxs, parent_shape, base=parent_base.itemsize)
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
				preceeding_fields = takewhile(lambda x: x != first_field, parent.dtype.base.names)
				self.offset = sum(parent_base[name].itemsize for name in preceeding_fields)
		self.offset += getattr(parent, 'offset', 0)

	def map(self, access=(GL.GL_MAP_READ_BIT | GL.GL_MAP_WRITE_BIT )):
		"""Returns a numpy array mapped to the sub-buffer.

		:param access: The OpenGL flags passed to map buffer range.

		.. admonition:: |buffer-bind|

		   This action binds the buffer.
		"""
		with self.buffer:
			mem = GL.glMapBufferRange(self.buffer.target, self.offset, self.dtype.itemsize, access)
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
	def buffer(self):
		"""The :py:class:`Buffer` object that this sub-buffer indexes."""
		parent = self.parent
		while hasattr(parent, 'parent'):
			parent = parent.parent
		return parent

	@property
	def nbytes(self):
		return self.dtype.itemsize

	@property
	def data(self):
		"""See :py:meth:`Buffer.data`"""
		with self.buffer:
			a = GL.glGetBufferSubData(self.buffer.target, self.offset, self.nbytes)
		a.dtype = self.dtype.base
		a.shape = self.dtype.shape
		return a

	@data.setter
	def data(self, value):
		value.dtype = self.dtype.base
		value.shape = self.dtype.shape
		with self.buffer:
			GL.glBufferSubData(self.buffer.target, self.offset, value.nbytes, value)

class BufferMapping(ndarray):
	"""A numpy array that is mapped to a buffer.

	Care must be taken not to access this array after it has been un-mapped.

	.. warning:: Undefined behaviour

	   Behavious is undefined if the array is used in a manner inconsistent with the flags it
	   was mapped with (e.g. reading an array mapped only with :py:obj:`GL.GL_MAP_WRITE_BIT`
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

		Otherwise, a memory barrier is issued.

		.. admonition:: |buffer-bind|

		   This action binds the buffer *only if* :py:obj:`GL.GL_MAP_FLUSH_EXPLICIT_BIT` was set
		   when the buffer was bound.
		"""
		if self.access & GL.GL_MAP_FLUSH_EXPLICIT_BIT:
			length = self.dtype.itemsize * product(self.shape)
			with self.gl_buffer:
				GL.glFlushMappedBufferRange(self.gl_buffer.target, self.offset, length)
		else:
			GL.glMemoryBarrier(GL.GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT)
