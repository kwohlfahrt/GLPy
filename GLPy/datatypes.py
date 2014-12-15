from OpenGL import GL
from OpenGL.constants import GLboolean,GLint, GLuint, GLfloat, GLdouble
from itertools import product as cartesian
# TODO: dobule-based types

from numpy import dtype

from util.misc import product
from enum import Enum

scalar_types = ['bool', 'int', 'uint', 'float', 'double']
class Scalar(Enum):
	'''The basic GLSL scalars.

	Scalars define the following attributes:

	*prefix*
	  The prefix used for related types, e.g. ``'b'`` for ``Scalar.bool`` as a
	  3-vector of booleans is a **b**\ vec3
	*base_type*
	  The base type of a scalar is itself
	'''

	__prefixes__ = { 'bool': 'b'
	               , 'int': 'i'
	               , 'uint': 'u'
	               , 'float': ''
	               , 'double': 'd' }

	def __init__(self, value):
		self.prefix = self.__prefixes__[self.name]
		self.base_type = self
scalar_doc = Scalar.__doc__
Scalar = Enum('Scalar', scalar_types, type=Scalar)
Scalar.__doc__ = scalar_doc

floating_point_scalars = { Scalar.float, Scalar.double }

sampler_dims = range(1, 4)
sampler_data_types = {Scalar.float, Scalar.int, Scalar.uint}
sampler_types = [ "{}sampler{}D".format(scalar_type.prefix, ndim)
                  for scalar_type, ndim in cartesian(sampler_data_types, sampler_dims) ]
class Sampler(Enum):
	__ndims__ = { "{}sampler{}D".format(scalar_type.prefix, ndim): ndim
	              for scalar_type, ndim in cartesian(sampler_data_types, sampler_dims) }

	def __init__(self, value):
		self.ndim = self.__ndims__[self.name]
Sampler = Enum('Sampler', sampler_types, type=Sampler)

vector_sizes = range(2, 5)
vector_types = ["{}vec{}".format(scalar_type.prefix, size)
                for scalar_type, size in cartesian(Scalar, vector_sizes) ]
class Vector(Enum):
	'''The GLSL vector types.

	Vectors define the following attributes:

	*base_type*
	  The :py:class:`Scalar` type that defines a single element of the vector
	*shape*
	  A 1-tuple of the number of elements in the vector
	'''
	__base_types__ = { "{}vec{}".format(scalar_type.prefix, size): scalar_type
	                    for scalar_type, size in cartesian(Scalar, vector_sizes) }
	__shapes__ = { "{}vec{}".format(scalar_type.prefix, size): (size,)
	              for scalar_type, size in cartesian(Scalar, vector_sizes) }

	def __init__(self, value):
		self.base_type = self.__base_types__[self.name]
		self.shape = self.__shapes__[self.name]
vector_doc = Vector.__doc__
Vector = Enum('Vector', vector_types, type=Vector)
Vector.__doc__ = vector_doc

matrix_types = ( ["{}mat{}".format(scalar_type.prefix, size)
                  for scalar_type, size in cartesian(floating_point_scalars, vector_sizes)]
               + ["{}mat{}x{}".format(scalar_type.prefix, size1, size2)
                  for scalar_type, size1, size2
                  in cartesian(floating_point_scalars, vector_sizes, vector_sizes)] )
class Matrix(Enum):
	'''The GLSL matrix types.

	Matrices define the following attributes:

	*base_type*
	  The :py:class:`Scalar` type that defines a single element of the matrix
	*shape*
	  A 2-tuple of the number of elements along each dimension
	'''

	__base_types__ = { "{}mat{}".format(scalar_type.prefix, size): scalar_type
	                   for scalar_type, size in cartesian(floating_point_scalars, vector_sizes) }
	__base_types__.update({ "{}mat{}x{}".format(scalar_type.prefix, size1, size2): scalar_type
	                        for scalar_type, size1, size2
	                        in cartesian(floating_point_scalars, vector_sizes, vector_sizes) })

	__shapes__ = { "{}mat{}".format(scalar_type.prefix, size): (size, size)
	              for scalar_type, size in cartesian(floating_point_scalars, vector_sizes) }
	__shapes__.update({ "{}mat{}x{}".format(scalar_type.prefix, size1, size2): (size1, size2)
	                   for scalar_type, size1, size2
	                   in cartesian(floating_point_scalars, vector_sizes, vector_sizes) })

	def __init__(self, value):
		self.shape = self.__shapes__[self.name]
		self.base_type = self.__base_types__[self.name]
matrix_doc = Matrix.__doc__
Matrix = Enum('Matrix', matrix_types, type=Matrix)
Matrix.__doc__ = matrix_doc

glsl_types = [Scalar, Vector, Matrix, Sampler]

class Struct:
	'''A GLSL ``struct``

	:param str name: The name of the struct
	:param \\*contents: The contents of the struct
	:type \\*contents: [:py:class:`.Variable`]
	:raises TypeError: If ``contents`` contains any opaque types.
	'''
	def __init__(self, name, *contents):
		self.name = name
		if not all(hasattr(c, base_type) for c in contents):
			raise TypeError("Interface blocks may not contain opaque types.")
		self.contents = contents

	def __str__(self):
		contents = '; '.join(str(c) for c in self.contents)
		return "struct {} {{ {}; }}".format(self.name, contents)

	def __repr__(self):
		return "{}(name='{}' contents={})".format(type(self).__name__, self.name, self.contents)

class Type:
	'''A GLSL type, not necessarily a named variable.

	:param gl_type: The OpenGL data type (e.g. ``vec3``)
	:type gl_type: :py:class:`.Scalar`, :py:class:`.Vector`
	  :py:class:`.Matrix`, :py:class:`.Sampler` or :py:class:`.Struct` or
	  :py:obj:`str`
	:param shape: The shape of the data type if it is an array
	:type shape: :py:obj:`int` or [:py:obj:`int`]
	'''
	def __init__(self, gl_type, shape=1):
		for glsl_type in glsl_types:
			try:
				self.type = glsl_type[gl_type]
				break
			except KeyError:
				pass
		else:
			self.type = gl_type
		try:
			self.shape = tuple(shape)
		except TypeError:
			self.shape = (shape,)
	
	def __str__(self):
		array = ''.join("[{}]".format(s) for s in self.shape) if self.shape != (1,) else ""
		return "{}{}".format(self.type, array)
	
	def __repr__(self):
		return "{}({})".format(type(self).__name__, self)
	
	@property
	def count(self):
		'''The total number of elements in the variable'''
		return product(self.shape)

class Variable(Type):
	'''A class to represent a named GLSL variable.

	:param str name: The name of the GLSL variable
	:param str gl_type: The GLSL data type (e.g. ``vec3``)
	:param shape: The shape of the data type if it is an array
	:type shape: :py:obj:`int` or [:py:obj:`int`]
	'''

	def __init__(self, name, gl_type, shape=1):
		super().__init__(gl_type=gl_type, shape=shape)
		self.name = name

	@classmethod
	def fromType(cls, name, glsl_type):
		'''Construct from a name and an instance of :py:class:`.Type`

		:param str name: the name of the variable
		:param glsl_type: The type of the variable
		:type glsl_type: :py:class:`.Type`
		'''
		return cls(name, glsl_type.type, glsl_type.shape)

	def __str__(self):
		array = ''.join("[{}]".format(s) for s in self.shape) if self.shape != (1,) else ""
		return "{} {}{}".format(self.type, self.name, array)

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
