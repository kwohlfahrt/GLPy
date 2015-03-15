from itertools import chain
from itertools import product as cartesian

from .datatypes import Scalar, Vector, Matrix, Array, Struct, BasicType

class Variable:
	'''A class to represent a named GLSL variable.

	:param str name: The name of the GLSL variable
	:param datatype: The GLSL data type, strings may be substitued for basic types (e.g. ``vec3``)
	:type datatype: :py:class:`Scalar`, :py:class:`Sampler`, :py:class:`Vector`, :py:class:`Matrix`,
	  :py:class:`Struct`, :py:class:`Array` or :py:obj:`str`
	'''

	def __init__(self, name, datatype):
		try:
			self.datatype = BasicType(datatype)
		except ValueError:
			self.datatype = datatype
		self.name = name

	def __repr__(self):
		return "<Variable name={} type={}>".format(self.name, self.datatype)

	def __str__(self):
		try:
			base = self.datatype.name
		except AttributeError:
			base = str(self.datatype)
		return ' '.join((base, self.name))

	def __eq__(self, other):
		return self.name == other.name and self.datatype == other.datatype

	def __hash__(self):
		return hash((self.name, self.datatype))

	def __getitem__(self, idx):
		'''Return a member or element of this variable if it is of an indexable datatype.

		:rtype: :py:class:`.Variable`

		:raises TypeError: If the Variable is not of an indexable datatype.
		'''
		if isinstance(self.datatype, Array):
			name = "{}[{}]".format(self.name, idx)
			return Variable(name, self.datatype[idx])
		elif isinstance(self.datatype, Struct):
			member = self.datatype[idx]
			name = '.'.join((self.name, member.name))
			return Variable(name, member.datatype)
		else:
			raise TypeError("{} is a basic type and cannot be indexed.".format(self.datatype))

	def __len__(self):
		return len(self.datatype)

	def __iter__(self):
		'''Iterate over the members or elements of this variable if it is of an iterable datatype.

		:rtype: [:py:class:`.Variable`]

		:raises TypeError: If the Variable is not of an indexable datatype.
		'''
		if isinstance(self.datatype, Array):
			for idx, element_type in enumerate(self.datatype):
				name = "{}[{}]".format(self.name, idx)
				yield Variable(name, element_type)
		elif isinstance(self.datatype, Struct):
			for member in self.datatype:
				name = '.'.join((self.name, member.name))
				yield Variable(name, member.datatype)
		else:
			raise TypeError("{} is a basic type and cannot be iterated over.".format(self.datatype))

	@property
	def resources(self):
		'''The resources that would be defined by this variable, assuming it is active

		:rtype: [:py:class:`Variable`] where the type of each variable is :py:class:`BasicType`, or
		  a :py:class:`Array` of basic types.
		'''
		if isinstance(self.datatype, Array):
			if isinstance(self.datatype.element, BasicType):
				return [Variable(''.join((self.name, '[0]')), self.datatype)]
			else:
				return list(chain.from_iterable(v.resources for v in self))
		elif isinstance(self.datatype, Struct):
			return list(chain.from_iterable(v.resources for v in self))
		else:
			return [self]
