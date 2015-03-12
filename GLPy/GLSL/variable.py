from itertools import chain
from itertools import product as cartesian

from .datatypes import Scalar, Vector, Matrix, Array, Struct, BasicType

class Variable:
	'''A class to represent a named GLSL variable.

	:param str name: The name of the GLSL variable
	:param gl_type: The GLSL data type, strings may be substitued for basic types (e.g. ``vec3``)
	:type gl_type: :py:class:`Scalar`, :py:class:`Sampler`, :py:class:`Vector`, :py:class:`Matrix`,
	  :py:class:`Struct`, :py:class:`Array` or :py:obj:`str`
	'''

	def __init__(self, name, gl_type):
		try:
			self.type = BasicType(gl_type)
		except ValueError:
			self.type = gl_type
		self.name = name

	def __repr__(self):
		return "<Variable name={} type={}>".format(self.name, self.type)

	def __str__(self):
		try:
			base = self.type.name
		except AttributeError:
			base = str(self.type)
		return ' '.join((base, self.name))

	def __eq__(self, other):
		return self.name == other.name and self.type == other.type

	def __hash__(self):
		return hash((self.name, self.type))

	def __getitem__(self, idx):
		if isinstance(self.type, Array):
			name = "{}[{}]".format(self.name, idx)
			return Variable(name, self.type[idx])
		elif isinstance(self.type, Struct):
			member = self.type[idx]
			name = '.'.join((self.name, member.name))
			return Variable(name, member.type)
		else:
			raise TypeError("{} is a basic type and cannot be indexed.".format(self.type))

	def __len__(self):
		return len(self.type)

	def __iter__(self):
		if isinstance(self.type, Array):
			for idx, element_type in enumerate(self.type):
				name = "{}[{}]".format(self.name, idx)
				yield Variable(name, element_type)
		elif isinstance(self.type, Struct):
			for member in self.type:
				name = '.'.join((self.name, member.name))
				yield Variable(name, member.type)
		else:
			raise TypeError("{} is a basic type and cannot be iterated over.".format(self.type))

	@property
	def resources(self):
		'''The resources that would be defined by this variable, assuming it is active

		:returns: The resources that would be defined by this variable.
		:rtype: [:py:class:`Variable`] where the type of each variable is :py:class:`BasicType`.
		'''
		if isinstance(self.type, Array):
			if isinstance(self.type.element, BasicType):
				return [Variable(''.join((self.name, '[0]')), self.type)]
			else:
				return list(chain.from_iterable(v.resources for v in self))
		elif isinstance(self.type, Struct):
			return list(chain.from_iterable(v.resources for v in self))
		else:
			return [self]
