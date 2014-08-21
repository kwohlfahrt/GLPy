from itertools import chain

from OpenGL import GL
import numpy

from .datatypes import sampler_types, gl_types

def clampSlice(s, size):
	r = [0, size]
	for i, v in enumerate((s.start, s.stop)):
		if v is None:
			continue
		r[i] = v if v > 0 else v + size
	return slice(*r)

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
	def __init__(self, size, components=4, count=1, levels=1
	            , bits=8, integer=True, normalized=True, signed=False
				, handle=None, **tex_params):
		self.bound = 0

		self.handle = handle or GL.glGenTextures(1)
		self.size = size
		self.levels = levels
		self.count = count
		self.components = components

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
		return getattr(GL, 'GL_TEXTURE_{}D{}'.format(len(self.size), '_ARRAY' if self.count > 1 else ''))
	
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
		if not self.bound:
			GL.glBindTexture(self.target, self.handle)
		self.bound += 1
	
	def __exit__(self, ex, val, tb):
		self.bound -= 1
		if not self.bound:
			GL.glBindTexture(self.target, 0)
	
	def __getitem__(self, idxs):
		raise NotImplementedError("Texture data access is not yet implemented.")
	
	def __setitem__(self, idxs, value):
		value = numpy.asarray(value)
		idxs = [idx if isinstance(idx, slice) else slice(idx, idx+1) for idx in idxs]
		if any(idx.step is not None for idx in idxs):
			raise IndexError("Cannot set discontinuous sections of a texture.")

		if len(idxs) == len(self.size):
			count = slice(None, self.count)
			components = slice(None, self.components)
			size = idxs
		elif len(idxs) == len(self.size) + 1:
			if self.count == 1:
				components = idxs[-1]
				size = idxs[:-1]
				count = slice(None, self.count)
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

		components = clampSlice(components, self.components)
		if components.start != 0:
			raise IndexError("Can only set components starting from 0.")
		if components.stop == 0:
			return
		if components.stop > self.components:
			raise IndexError("Cannot set more components than the texture has.")
		value.dtype = numpy.dtype(value.dtype.base, components.stop)

		count = clampSlice(count, self.count)
		for i, (s, size_dim) in enumerate(zip(size, self.size)):
			size[i] = clampSlice(s, size_dim)
		if count.start < 0:
			raise IndexError("Cannot set before start of array.")
		if count.stop > self.count:
			raise IndexError("Cannot set beyond end of array.")
		new_shape = [count.stop - count.start] + [s.stop - s.start for s in size]
		if any(s <= 0 for s in new_shape):
			return
		value.shape = new_shape if self.count > 1 else new_shape[1:]

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
		if any(u <= 0 for u in units):
			raise ValueError("Cannot bind to texture units under 1.")
		for u in units:
			GL.glActiveTexture(GL.GL_TEXTURE0 + u)
			GL.glBindTexture(self.target, self.handle)
		GL.glActiveTexture(GL.GL_TEXTURE0)
		GL.glBindTexture(self.target, 0)
