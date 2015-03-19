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
		base_type = getattr(self.datatype, 'base', self.datatype)
		if not isinstance(base_type, BasicType):
			raise TypeError("Invalid type for a vertex attribute.")
		self.shader_location = location
		'''The location of this attribute if specified in the shader'''
		self.normalized = normalized
		'''Whether a floating-point data type should be normalized from integer data'''

	def __eq__(self, other):
		return ( super().__eq__(other) and self.shader_location == other.shader_location
		       and self.normalized == other.normalized )

	def __repr__(self):
		return ("<VertexAttribute name={}, datatype={}, location={}, normalized={}>"
		        .format(self.name, self.datatype, self.shader_location, self.normalized))

	def __str__(self):
		base = super().__str__()
		if self.shader_location is not None:
			layout = "layout(location={})".format(self.location)
			return ' '.join((layout, 'in', base))
		return ' '.join(('in', base))

	def __getitem__(self, idx):
		var = super().__getitem__(idx)

		array_elements = product(getattr(var.datatype, 'full_shape', (1,)))
		base_type = getattr(var.datatype, 'base', var.datatype)
		element_indices = getattr(base_type, 'columns', 1)

		if self.shader_location is None:
			location = None
		else:
			location = self.shader_location + idx * element_indices * array_elements
		return VertexAttribute(var.name, var.datatype, location, self.normalized)
