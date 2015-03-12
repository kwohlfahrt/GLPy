from itertools import repeat, product as cartesian
from collections import OrderedDict

from numpy import dtype

from enum import Enum

class BasicType:
	'''A base class for all GLSL basic types:

	- :py:class:`Scalar`
	- :py:class:`Vector`
	- :py:class:`Matrix`
	- :py:class:`Sampler`

	It supports construction from a string representation of the GLSL type for convenience:

	>>> BasicType('vec3') is Vector.vec3
	True
	'''

	def __new__(self, gl_type):
		for basic_type in [Scalar, Vector, Matrix, Sampler]:
			try:
				return basic_type[gl_type]
			except KeyError:
				pass
		else:
			raise ValueError("No such GLSL type.")

scalar_types = ['bool', 'int', 'uint', 'float', 'double']

class Scalar(str, BasicType, Enum):
	'''The basic GLSL scalars.

	Scalars define the following attributes:

	*prefix*
	  The prefix used for related types, e.g. ``'b'`` for ``Scalar.bool`` as a
	  3-vector of booleans is a **b**\ vec3
	*scalar_type*
	  The scalar type of a scalar is itself
	*machine_type*
	  The machine representation of this GLSL type as a :py:class:`numpy.dtype`
	'''

	__prefixes__ = { 'bool': 'b'
	               , 'int': 'i'
	               , 'uint': 'u'
	               , 'float': ''
	               , 'double': 'd' }
	__machine_types__ = {'bool': dtype('uint32')
						,'int': dtype('int32')
						,'uint': dtype('uint32')
						,'float': dtype('float32')
						,'double': dtype('float64')}

	def __init__(self, value):
		self.prefix = self.__prefixes__[self.name]
		self.machine_type = self.__machine_types__[self.name]
		self.scalar_type = self
		self.opaque = False
scalar_doc = Scalar.__doc__
Scalar = Enum('Scalar', ((s, s) for s in scalar_types), type=Scalar)
Scalar.__doc__ = scalar_doc

floating_point_scalars = { Scalar.float, Scalar.double }

sampler_dims = range(1, 4)
sampler_data_types = {Scalar.float, Scalar.int, Scalar.uint}
sampler_types = [ "{}sampler{}D".format(scalar_type.prefix, ndim)
                  for scalar_type, ndim in cartesian(sampler_data_types, sampler_dims) ]
class Sampler(str, BasicType, Enum):
	__ndims__ = { "{}sampler{}D".format(scalar_type.prefix, ndim): ndim
	              for scalar_type, ndim in cartesian(sampler_data_types, sampler_dims) }

	def __init__(self, value):
		self.ndim = self.__ndims__[self.name]
		self.opaque = True
Sampler = Enum('Sampler', ((s, s) for s in sampler_types), type=Sampler)

vector_sizes = range(2, 5)
vector_types = ["{}vec{}".format(scalar_type.prefix, size)
                for scalar_type, size in cartesian(Scalar, vector_sizes) ]
class Vector(str, BasicType, Enum):
	'''The GLSL vector types.

	Vectors define the following attributes:

	*scalar_type*
	  The :py:class:`Scalar` type that defines a single element of the vector
	*shape*
	  A 1-tuple of the number of elements in the vector
	*machine_type*
	  The machine representation of this GLSL type as a :py:class:`numpy.dtype`
	'''
	__scalar_types__ = { "{}vec{}".format(scalar_type.prefix, size): scalar_type
	                     for scalar_type, size in cartesian(Scalar, vector_sizes) }
	__shapes__ = { "{}vec{}".format(scalar_type.prefix, size): (size,)
	              for scalar_type, size in cartesian(Scalar, vector_sizes) }

	def __init__(self, value):
		self.scalar_type = self.__scalar_types__[self.name]
		self.shape = self.__shapes__[self.name]
		self.machine_type = dtype((self.scalar_type.machine_type, self.shape))
		self.opaque = False

	@classmethod
	def fromType(cls, scalar_type, size):
		return cls[''.join((scalar_type.prefix, 'vec', str(size)))]

vector_doc = Vector.__doc__
Vector = Enum('Vector', ((v, v) for v in vector_types), type=Vector)
Vector.__doc__ = vector_doc

matrix_types = ( ["{}mat{}".format(scalar_type.prefix, size)
                  for scalar_type, size in cartesian(floating_point_scalars, vector_sizes)]
               + ["{}mat{}x{}".format(scalar_type.prefix, size1, size2)
                  for scalar_type, size1, size2
                  in cartesian(floating_point_scalars, vector_sizes, vector_sizes)] )
class Matrix(str, BasicType, Enum):
	'''The GLSL matrix types.

	Matrices define the following attributes:

	*scalar_type*
	  The :py:class:`Scalar` type that defines a single element of the matrix
	*shape*
	  A 2-tuple of the number of elements along each dimension
	'''

	__scalar_types__ = { "{}mat{}".format(scalar_type.prefix, size): scalar_type
	                     for scalar_type, size in cartesian(floating_point_scalars, vector_sizes) }
	__scalar_types__.update({ "{}mat{}x{}".format(scalar_type.prefix, size1, size2): scalar_type
	                          for scalar_type, size1, size2
	                          in cartesian(floating_point_scalars, vector_sizes, vector_sizes) })

	__shapes__ = { "{}mat{}".format(scalar_type.prefix, size): (size, size)
	              for scalar_type, size in cartesian(floating_point_scalars, vector_sizes) }
	__shapes__.update({ "{}mat{}x{}".format(scalar_type.prefix, size1, size2): (size1, size2)
	                   for scalar_type, size1, size2
	                   in cartesian(floating_point_scalars, vector_sizes, vector_sizes) })

	def __init__(self, value):
		self.shape = self.__shapes__[self.name]
		self.scalar_type = self.__scalar_types__[self.name]
		self.machine_type = dtype((self.scalar_type.machine_type, self.shape))
		self.opaque = False

	@classmethod
	def fromType(cls, scalar_type, shape):
		columns, rows = shape
		return cls[''.join((scalar_type.prefix, 'mat', str(columns), 'x', str(rows)))]

	@property
	def rows(self):
		return self.shape[1]

	@property
	def columns(self):
		return self.shape[0]
matrix_doc = Matrix.__doc__
Matrix = Enum('Matrix', ((m, m) for m in matrix_types), type=Matrix)
Matrix.__doc__ = matrix_doc

glsl_types = [Scalar, Vector, Matrix, Sampler]

class Struct:
	'''A GLSL ``struct``

	:param str name: The name of the struct
	:param \\*contents: The contents of the struct
	:type \\*contents: [:py:class:`.Variable`]
	'''

	def __init__(self, name, *contents):
		self.name = name
		self.contents = OrderedDict((var.name, var) for var in contents)

	def __str__(self):
		contents = '; '.join(str(c) for c in self.contents)
		return "struct {} {{ {}; }}".format(self.name, contents)

	def __repr__(self):
		return "{}(name='{}' contents={})".format(type(self).__name__, self.name, self.contents)

	def __len__(self):
		return len(self.contents)

	def __getitem__(self, idx):
		return self.contents[idx]

	def __iter__(self):
		return iter(self.contents.values())

	def __hash__(self):
		return hash((self.name, tuple(self.contents.items())))

	def __eq___(self):
		return self.name == other.name and self.contents == other.contents

def formatShape(shape):
	array = ']['.join(str(s) for s in shape)
	return '[{}]'.format(array)

class Array:
	'''A GLSL array.

	:param element: The OpenGL type of one element of this array.
	:type element: :py:class:`.Scalar`, :py:class:`.Vector`
	  :py:class:`.Matrix`, :py:class:`.Sampler` or :py:class:`.Struct`,
	  :py:class:`.Array` or :py:obj:`str`
	:param shape: The shape of the array. A sequence will be transformed into
	  an array of arrays.
	:type shape: :py:obj:`int` or [:py:obj:`int`]
	'''
	def __init__(self, base, shape=1):
		try:
			base = BasicType(base)
		except ValueError:
			pass

		try:
			shape, *child_shapes = shape
		except TypeError:
			child_shapes = False

		# Distinguish from 'Vector' and 'Matrix' shapes
		self.array_shape = shape
		self.element = base if not child_shapes else Array(base, child_shapes)

	@property
	def full_shape(self):
		'''The shape of this array and all child arrays.'''
		return (self.array_shape, ) + getattr(self.element, 'full_shape', ())

	@property
	def base(self):
		'''The non-array base of this array.'''
		return getattr(self.element, 'base', self.element)

	def __str__(self):
		return ''.join((self.base.name, formatShape(self.full_shape)))

	def __getitem__(self, idx):
		if not 0 <= idx < self.array_shape:
			raise IndexError("No such array element '{}'".format(idx))
		return self.element # All elements identical

	def __len__(self):
		return self.array_shape

	def __iter__(self):
		return iter(repeat(self.element, self.array_shape))

	def __eq__(self, other):
		return self.element == other.element and self.array_shape == other.array_shape

	def __hash__(self):
		return hash((self.element, self.array_shape))
