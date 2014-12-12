from OpenGL import GL
from OpenGL.constants import GLboolean, GLint, GLuint, GLfloat, GLdouble
from numpy import dtype

from collections import namedtuple

from util.misc import product
from .datatypes import Scalar, Vector, Matrix

numpy_buffer_types = { dtype('int8'): GL.GL_BYTE
                     , dtype('uint8'): GL.GL_UNSIGNED_BYTE
                     , dtype('int16'): GL.GL_SHORT
                     , dtype('uint16'): GL.GL_UNSIGNED_SHORT
                     , dtype('int32'): GL.GL_INT
                     , dtype('uint32'): GL.GL_UNSIGNED_INT
                     , dtype('float16'): GL.GL_HALF_FLOAT
                     , dtype('float32'): GL.GL_FLOAT
                     , dtype('float64'): GL.GL_DOUBLE }
integer_buffer_types = { GL.GL_BYTE, GL.GL_UNSIGNED_BYTE
                       , GL.GL_SHORT, GL.GL_UNSIGNED_SHORT
                       , GL.GL_INT, GL.GL_UNSIGNED_INT }
floating_point_buffer_types = { GL.GL_HALF_FLOAT, GL.GL_FLOAT, GL.GL_DOUBLE }

Empty = namedtuple("Empty", ["nbytes"])
Empty.__doc__ = """A class to represent an empty buffer

                :param int nbytes: Size of the empty buffer"""

class Buffer:
	"""An OpenGL buffer. Needs to be sub-classed and define the ``target`` property for use.

	:keyword usage: The intended usage of the buffer.
	:keyword handle: The buffer handle. One will be created if it is not provided.
	:type handle: :py:obj:`int` or :py:obj:`None`
	"""

	def __init__(self, usage=GL.GL_DYNAMIC_DRAW, handle=None):
		self.usage = usage
		self.handle = GL.glGenBuffers(1) if handle is None else handle
		self.bound = 0
		self.bytes = BufferBytes(self)
	
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

class BufferBytes(Buffer):
	"""A class used to represent the internal storage of a buffer object.  Should not be
	instantiated directly.
	"""

	def __init__(self, buf):
		self.buf = buf
		self.nbytes = None
	
	def __len__(self):
		"""The length of the buffer.

		:return: The length of the buffer (in bytes), or :py:obj:`None` if it is not allocated
		:rtype: :py:obj:`int` or :py:obj:`None`
		"""
		return self.nbytes
	
	def __getitem__(self, i):
		raise NotImplementedError("Buffer data access is not yet implemented.")
	
	def __setitem__(self, i, value):
		"""Set the buffer data or sub-data.

		:param i: Section of the buffer to set, in bytes. Passing :py:obj:`Ellipsis` will resize the
			buffer to fit the data
		:type i: :py:obj:`slice` or :py:obj:`int` or :py:obj:`Ellipsis`
		:param value: New data
		:type value: :py:class:`numpy.ndarray` or :py:class:`.Empty`
		:raises IndexError: IndexError will be raised if ``i`` does not represent a contiguous
			buffer section
		:raises ValueError: ValueError will be raised if ``value`` is larger than the slice ``i``

		.. admonition:: |buffer-bind|
		
		   This method binds the buffer the bytes belong to."""

		if i is Ellipsis:
			size = value.nbytes
			if isinstance(value, Empty):
				value = None
			with self.buf:
				GL.glBufferData(self.buf.target, size, value, self.buf.usage)
				self.nbytes = size
			return

		if not isinstance(i, slice):
			i = slice(i, i + 1)
		if i.step not in (None, 1):
			raise IndexError("Cannot set non-contiguous buffer data.")
		i = range(*i.indices(self.nbytes))

		if (i.stop - i.start) != value.nbytes:
			raise ValueError("Data size does not match  buffer range.")
		with self.buf:
			GL.glBufferSubData(self.buf.target, i.start, len(i), value)
