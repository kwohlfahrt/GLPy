from itertools import chain

from OpenGL import GL
import numpy

from .datatypes import sampler_types, gl_types

create_storage = {getattr(GL, 'GL_TEXTURE_{}D'.format(d)):
                  getattr(GL, 'glTexStorage{}D'.format(d))
                  for d in range(1, 4)}
create_storage.update({getattr(GL, 'GL_TEXTURE_{}D_ARRAY'.format(d)):
                       getattr(GL, 'glTexStorage{}D'.format(d + 1))
                       for d in range(1, 3)})

set_sub_texture = {getattr(GL, 'GL_TEXTURE_{}D'.format(d)):
                   getattr(GL, 'glTexSubImage{}D'.format(d))
                   for d in range(1, 4)}
set_sub_texture.update({getattr(GL, 'GL_TEXTURE_{}D_ARRAY'.format(d)):
                        getattr(GL, 'glTexSubImage{}D'.format(d + 1))
                        for d in range(1, 3)})

set_texture = {getattr(GL, 'GL_TEXTURE_{}D'.format(d)):
               getattr(GL, 'glTexImage{}D'.format(d))
               for d in range(1, 4)}
set_texture.update({getattr(GL, 'GL_TEXTURE_{}D_ARRAY'.format(d)):
                    getattr(GL, 'glTexImage{}D'.format(d + 1))
                    for d in range(1, 3)})

class ImmutableTexture:
	'''A class for textures (using immutable storage). These textures are
	created with an explicit size, type and number of mipmap levels, and cannot
	be resized.
	
	:param [int] size: The size of the texture
	:param int components: Number of components per pixel
	:param count: Number of slices (will create an array texture if this is not
		None)
	:type count: int or None
	:param int levels: Number of mipmap levels. Only ``1`` is currently
		supported.
	:param int bits: Bits per pixel
	:param bool integer: Use an integer texture type (instead of a float)
	:param bool normalized: Create a normalized texture (if integer)
	:param bool signed: Create a signed texture (if integer)
	:param handle: The OpenGL handle to use for the texture
	:type handle: int or None
	:param \*\*tex_params: The texture parameters to be set'''

	def __init__(self, size, components=4, count=None, levels=1
	            , bits=8, integer=True, normalized=True, signed=False
				, handle=None, **tex_params):
		self.bound = 0

		self.handle = handle or GL.glGenTextures(1)
		self.levels = levels
		self.count = count
		self.components = components
		try:
			self.size = tuple(size)
		except TypeError:
			self.size = (size,)

		self.integer = integer
		self.signed = signed
		self.normalized = normalized
		self.bits = bits

		for param, value in tex_params.items():
			pname = getattr(GL, param)
			glTexParameterfv(self.handle, pname, value)

		with self:
			create_storage[self.target](self.target, self.levels, self.internal_format, *self.size)
	
	@property
	def target(self):
		return getattr(GL, 'GL_TEXTURE_{}D{}'.format(len(self.size), '_ARRAY' if self.count is not None else ''))
	
	@property
	def internal_format(self):
		if self.integer:
			if self.normalized:
				if self.signed:
					internal_type = '_SNORM'
				else:
					internal_type = ''
			else:
				if self.signed:
					internal_type = 'I'
				else:
					internal_type = 'UI'
		else:
			internal_type = 'F'

		format_str = 'GL_{}{}{}'.format('RGBA'[:self.components], self.bits, internal_type)
		return getattr(GL, format_str)
	
	def __enter__(self):
		"""Texture objects provide a context manager. This keeps track of how
		many times the texture has been bound and unbound. Grouping operations
		on a texture within a context where it is bound prevents repeated
		binding and un-binding.

		.. _texture-bind-warning:
		.. warning::
		   It is not allowed to bind two textures with the same target
		   simultaneously. It is allowed to bind the *same* texture multiple times.
		   
		   Methods that bind a texture will be documented"""
		if not self.bound:
			GL.glBindTexture(self.target, self.handle)
		self.bound += 1
	
	def __exit__(self, ex, val, tb):
		self.bound -= 1
		if not self.bound:
			GL.glBindTexture(self.target, 0)
	
	def __getitem__(self, idxs):
		raise NotImplementedError("Texture data access is not yet implemented.")
	
	# TODO: Allow setting of multiple mipmap levels
	def __setitem__(self, idxs, value):
		"""Set the image data. The behaviour changes depending on how many
		indices are provided and the type of the texture:

		- Equal to the number of dimensions: All components and array elements
		  are set.
		- One greater than the number of dimensions:
			- Array texture: The first index defines which array elements are set
			- Simple texture: The last index defines which components are set
		- Two greater than the number of dimensions: The first index defines
		  which array elements are set, the last index defines which components
		  are set.

		:param idxs: The indices to be set.
		:param value: The new image data."""
		value = numpy.asarray(value)
		idxs = [idx if isinstance(idx, slice) else slice(idx, idx+1) for idx in idxs]
		if any(idx.step is not None for idx in idxs):
			raise IndexError("Cannot set discontinuous sections of a texture.")

		tex_count = 1 if self.count is None else self.count

		if len(idxs) == len(self.size):
			count = slice(None, tex_count)
			components = slice(None, self.components)
			size = idxs
		elif len(idxs) == len(self.size) + 1:
			if tex_count is None:
				components = idxs[-1]
				size = idxs[:-1]
				count = slice(None, tex_count)
			else:
				components = slice(None, self.components)
				count = idxs[0]
				size = idxs[1:]
		elif len(idxs) == len(self.size) + 2:
			components = idxs[-1]
			count = idxs[0]
			size = idxs[1:-1]
		else:
			raise IndexError("Incorrect number of indices.")

		# FIXME: RED, GREEN and BLUE are all valid set components
		components = slice(*components.indices(self.components))
		if components.start != 0:
			raise IndexError("Can only set components starting from 0.")
		if components.stop == 0:
			return
		if components.stop > self.components:
			raise IndexError("Cannot set more components than the texture has.")
		value.dtype = numpy.dtype(value.dtype.base, components.stop)

		count = slice(*count.indices(tex_count))
		for i, (s, size_dim) in enumerate(zip(size, self.size)):
			size[i] = slice(*s.indices(size_dim))
		if count.start < 0:
			raise IndexError("Cannot set before start of array.")
		if count.stop > tex_count:
			raise IndexError("Cannot set beyond end of array.")
		new_shape = [count.stop - count.start] + [s.stop - s.start for s in size]
		if any(s <= 0 for s in new_shape):
			return
		value.shape = new_shape if tex_count > 1 else new_shape[1:]

		if self.integer and not self.normalized:
			value_format = '_'.join(('GL', 'RED' if components.stop == 1 else 'RGBA'[components], 'INTEGER'))
		else:
			value_format = '_'.join(('GL', 'RED' if components.stop == 1 else 'RGBA'[components]))
			
		value_format = getattr(GL, value_format)
		value_type = gl_types[value.dtype.base]
		level = 0

		# UPSTREAM: PyOpenGL functions do not take keyword arguments
		args = tuple(i.start for i in reversed(size)) + value.shape[::-1] + (value_format, value_type, value)
		with self:
			set_sub_texture[self.target](self.target, level, *args)
	
	def activate(self, *units):
		'''Binds the texture to the specified image units.

		.. admonition:: |texture-bind|

		  This method binds the texture it belongs to

		:param int \*units: The image units to bind to.
		:raises ValueError: if it is attempted to bind a texture to
			``GL_TEXTURE0``. This unit is reserved to permit safe unbinding of
			the texture from its target.'''

		if any(u <= 0 for u in units):
			raise ValueError("Cannot bind to texture units under 1.")
		for u in units:
			GL.glActiveTexture(GL.GL_TEXTURE0 + u)
			GL.glBindTexture(self.target, self.handle)
		GL.glActiveTexture(GL.GL_TEXTURE0)
		GL.glBindTexture(self.target, 0)
