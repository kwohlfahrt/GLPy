from OpenGL import GL

from .datatypes import gl_types

from util.misc import product

from collections import namedtuple

Empty = namedtuple("Empty", ["nbytes"])

class Buffer:
	"""An OpenGL buffer. Needs to be sub-classed and define the ``target``
	property for use.
	   
	.. _buffer-bind-warning:
	.. warning::
	   It is not allowed to bind two buffers with the same target
	   simultaneously. It is allowed to bind the *same* buffer multiple times.
	   
	   Methods that bind a buffer will be documented."""

	def __init__(self, usage=GL.GL_DYNAMIC_DRAW, handle=None):
		"""Creates a new buffer object, creating a new handle if None is provided.
		   
		:keyword usage: The intended usage of the buffer.
		:keyword handle: The buffer handle. One will be created if it is not provided.
		:type handle: int or None"""
		self.usage = usage
		self.handle = GL.glGenBuffers(1) if handle is None else handle
		self.bound = 0
		self.nbytes = None
		self.bytes = BufferBytes(self)
	
	def __enter__(self):
		"""Buffer objects provide a context manager. This keeps track of how
		many times the buffer has been bound and unbound. Grouping operations
		on a buffer within a context where it is bound prevents repeated
		binding and un-binding.
		
		.. admonition:: |buffer-bind|
		
		   This method binds the buffer it belongs to."""
		# ASSERT: self.bound >= 0
		if not self.bound:
			GL.glBindBuffer(self.target, self.handle)
		self.bound += 1

	def __exit__(self, ex, val, tr):
		# ASSERT: self.bound > 1
		self.bound -= 1
		if not self.bound:
			GL.glBindBuffer(self.target, 0)

class BufferBytes(Buffer):
	def __init__(self, buf):
		self.buf = buf
	
	def __len__(self):
		return self.buf.nbytes
	
	def __getitem__(self, i):
		raise NotImplementedError("Buffer data access is not yet implemented.")
	
	def __setitem__(self, i, value):
		"""Set the buffer data or sub-data.

		:param slice i: Section of the buffer to set, in bytes. ``i.step`` must be ``0`` or ``None``.
		:param value: New data, the buffer will be resized to fit if necessary.

		.. admonition:: |buffer-bind|
		
		   This method binds the buffer the bytes belong to."""
		if i == slice(None, None, None):
			size = value.nbytes
			if isinstance(value, Empty):
				value = None
			with self.buf:
				if size == self.buf.nbytes:
					GL.glBufferSubData(self.buf.target, 0, size, value)
				else:
					GL.glBufferData(self.buf.target, size, value, self.buf.usage)
					self.buf.nbytes = size
			return

		if not (i.step == 0 or i.step is None):
			raise ValueError("Cannot set non-contiguous buffer data.")
		with self.buf:
			# Rely on PyOpenGL to catch overflows
			GL.glBufferSubData(self.buf.target, i.start, i.stop - i.start, value)
