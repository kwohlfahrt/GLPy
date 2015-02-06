from OpenGL import GL
from OpenGL.constants import GLboolean,GLint, GLuint, GLfloat, GLdouble
from itertools import repeat, chain, product as cartesian
from collections import OrderedDict

from numpy import dtype

from util.misc import product
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
	'''

	__prefixes__ = { 'bool': 'b'
	               , 'int': 'i'
	               , 'uint': 'u'
	               , 'float': ''
	               , 'double': 'd' }

	def __init__(self, value):
		self.prefix = self.__prefixes__[self.name]
		self.scalar_type = self
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
	'''
	__scalar_types__ = { "{}vec{}".format(scalar_type.prefix, size): scalar_type
	                     for scalar_type, size in cartesian(Scalar, vector_sizes) }
	__shapes__ = { "{}vec{}".format(scalar_type.prefix, size): (size,)
	              for scalar_type, size in cartesian(Scalar, vector_sizes) }

	def __init__(self, value):
		self.scalar_type = self.__scalar_types__[self.name]
		self.shape = self.__shapes__[self.name]

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
		return iter(self.contents)

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
		self.element = Array(base, child_shapes) if child_shapes else base

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
			for idx in self.type:
				member = self.type[idx]
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

class BlockMemoryLayout(Enum):
	shared = 1
	packed = 2
	std140 = 3
	std430 = 4

class MatrixMemoryLayout(Enum):
	column_major = 1
	row_major = 2

# May have to separate out block and instance for different shader stages.
class InterfaceBlock:
	'''A generic interface block.

	Not to be instantiated directly, but as a base for defined block types.

	See :py:class:`InterfaceBlockMember` for additional exceptions that might be raised.
	
	:param str name: The name of the uniform block
	:param \\*members: The members of the uniform block. They may not contain
	  opaque types (e.g. :py:class:`.Sampler`)
	:type \\*members: :py:class:`.Variable`
	:param str instance_name: The name of the instance
	:param shape: The shape of the variable.
	:type shape: [:py:obj:`int`]
	:param layout: The layout of the interface block
	:raises ValueError: If an instance name is not defined and the block has a shape
	  larger than (1,)
	'''

	def __init__(self, name, *members, instance_name='', shape=1, layout='shared'):
		self.name = name
		self.members = [InterfaceBlockMember.fromVariable(self, m) for m in members]
		self.instance_name = instance_name
		try:
			self.shape = tuple(shape)
		except TypeError:
			self.shape = (shape,)
		if self.shape != (1,) and not self.instance_name:
			raise ValueError("An interface block may only be an array if it has an instance name.")
		self.layout = BlockMemoryLayout[layout]

	@property
	def dtype(self):
		if self.layout == BlockMemoryLayout.std140:
			raise NotImplementedError("TODO")
		elif self.layout == BlockMemoryLayout.std430:
			raise NotImplementedError("TODO")
		else:
			raise TypeError("The layout for this interface block is not defined.")

class InterfaceBlockMember(Variable):
	'''A variable that is a member of an interface block.

	Constructed implicitly from contents passed to a :py:class:`.InterfaceBlock`.

	:raises TypeError: If it is passed an opaque type as a base'''
	def __init__(self, block, name, gl_type, shape=1, matrixlayout='column_major'):
		super().__init__(name, gl_type, shape)
		if not hasattr(gl_type, 'base_type'):
			raise TypeError("Interface blocks may not contain opaque types.")
		if isinstance(self.type, Matrix):
			self.matrixlayout = MatrixMemoryLayout[matrixlayout]
		self.block = block

	@classmethod
	def fromVariable(cls, block, var, layout='column_major'):
		'''Construct from a block and a :py:class:`.Variable`

		:param str block: the block the variable belongs to
		:param var: the variable describing the block member
		:type var: :py:class:`.Variable`
		'''
		return cls(block, var.name, var.type, var.shape, layout)

	# TODO: array indices
	@property
	def gl_name(self):
		'''The string used to refer to the block member in a shader'''
		return (self.name if not self.block.instance_name
		        else '.'.join((self.block.instance_name, self.name)))

	@property
	def api_name(self):
		'''The string used to refer to the block member in the OpenGL API'''
		return (self.name if not self.block.instance_name
		        else '.'.join((self.block.name, self.name)))
