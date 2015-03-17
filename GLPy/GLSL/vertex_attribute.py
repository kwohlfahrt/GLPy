from itertools import repeat, chain
from .datatypes import Scalar, Vector, Matrix, BasicType, Array
from .variable import Variable

from util.misc import product

class VertexAttribute(Variable):
	"""A vertex attribute.

	:param location: The location of this vertex attribute, if defined explicitly in the shader.
	:type location: :py:obj:`int` or :py:obj:`None`
	:raises TypeError: If the passed GLSL type is not an instance of :py:class:`.BasicType` or an
	  :py:class:`.Array` of :py:class:`.BasicType`
	"""

	# How does glVertexAttribDivisor work with glVertexBindingDivisor?
	def __init__(self, name, datatype, location=None, normalized=False):
		super().__init__(name=name, datatype=datatype)
		# Could also check for base_type attribute
		base_type = getattr(self.datatype, 'base', self.datatype)
		if not isinstance(base_type, BasicType):
			raise TypeError("Invalid type for a vertex attribute.")
		self.shader_location = location
		'''The location of this attribute if specified in the shader'''
		self.normalized = normalized
		'''Whether a floating-point data type should be normalized from integer data'''
	
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
	def components(self):
		'''The number of components of a single attribute index of this type.'''

		datatype = getattr(self.datatype, 'base', self.datatype)
		return getattr(datatype, 'shape', (1,))[-1]
