from OpenGL import GL
from OpenGL.constants import GLboolean,GLint, GLuint, GLfloat, GLdouble
from itertools import product as cartesian
# TODO: dobule-based types

from numpy import dtype

from util.misc import product
from enum import Enum

class Scalar(Enum):
	bool = 1
	int = 2
	uint = 3
	float = 4
	double = 5

	__prefixes__ = { 'bool': 'b'
	               , 'int': 'i'
	               , 'uint': 'u'
	               , 'float': ''
	               , 'double': 'd' }

	def __init__(self, value):
		self.prefix = self.__prefixes__[self.name]
		self.base_type = self

floating_point_scalars = { Scalar.float, Scalar.double }

sampler_dims = range(1, 4)
sampler_data_types = {Scalar.float, Scalar.int, Scalar.uint}
sampler_types = [ "{}sampler{}D".format(scalar_type.prefix, ndim)
                  for scalar_type, ndim in cartesian(sampler_data_types, sampler_dims) ]
class Sampler(Enum):
	__sampler_ndim__ = { "{}sampler{}D".format(scalar_type.prefix, ndim): ndim
	                     for scalar_type, ndim in cartesian(sampler_data_types, sampler_dims) }

	def __init__(self, value):
		self.ndim = self.sampler__ndim__[self.name]
Sampler = Enum('Sampler', sampler_types)

vector_sizes = range(2, 5)
vector_types = ["{}vec{}".format(scalar_type.prefix, size)
                for scalar_type, size in cartesian(Scalar, vector_sizes) ]
class Vector(Enum):
	__base_types__ = { "{}vec{}".format(scalar_type.prefix, size): scalar_type
	                    for scalar_type, size in cartesian(Scalar, vector_sizes) }
	__shapes__ = { "{}vec{}".format(scalar_type.prefix, size): (size,)
	              for scalar_type, size in cartesian(Scalar, vector_sizes) }

	def __init__(self, value):
		self.base_type = self.__base_types__[self.name]
		self.shape = self.__shapes__[self.name]
Vector = Enum('Vector', vector_types, type=Vector)

matrix_types = ( ["{}mat{}".format(scalar_type.prefix, size)
                  for scalar_type, size in cartesian(floating_point_scalars, vector_sizes)]
               + ["{}mat{}x{}".format(scalar_type.prefix, size1, size2)
                  for scalar_type, size1, size2
                  in cartesian(floating_point_scalars, vector_sizes, vector_sizes)] )
class Matrix(Enum):
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

Matrix = Enum('Matrix', matrix_types, type=Matrix)

glsl_types = [Scalar, Vector, Matrix, Sampler]

#gl_types = {v: k for k, v in buf_types.items()}

class Struct:
	'''A GLSL ``struct``

	:param str name: The name of the struct
	:param \\*contents: The contents of the struct
	:type \\*contents: [:py:class:`.GLSLVar`]
	'''
	def __init__(self, name, *contents):
		self.name = name
		if not all(hasattr(c, base_type) for c in contents):
			raise ValueError("Interface blocks may not contain opaque types.")
		self.contents = contents

	def __str__(self):
		contents = '; '.join(str(c) for c in self.contents)
		return "struct {} {{ {}; }}".format(self.name, contents)

	def __repr__(self):
		return "{}(name='{}' contents={})".format(type(self).__name__, self.name, self.contents)

class Type:
	'''OpenGL type, not necessarily a named variable.

	:param gl_type: The OpenGL data type (e.g. ``vec3``)
	:type gl_type: :py:class:`.BaseType` or :py:class:`.Struct`
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
	'''A class to represent a named OpenGL variable.

	:param str name: The name of the OpenGL variable
	:param str gl_type: The OpenGL data type (e.g. ``vec3``)
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
	'''A generic interface block. Not to be instantiated directly, but as a
	base for defined block types.
	
	:param str name: The name of the uniform block
	:param \\*members: The members of the uniform block
	:type \\*members: :py:class:`.Variable`
	:param str instance_name: The name of the instance
	:param shape: The shape of the variable. Requires an instance name if not `1`
	:type shape: [:py:obj:`int`]
	:param layout: The layout of the interface block'''

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
	'''A variable that is a member of a data block (e.g. a uniform block)'''
	def __init__(self, block, name, gl_type, shape=1, matrixlayout='column_major'):
		super().__init__(name, gl_type, shape)
		if not hasattr(gl_type, 'base_type'):
			raise ValueError("Interface blocks may not contain opaque types.")
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
		'''The string used to refer to the block member in the OpenGL AP'''
		return (self.name if not self.block.instance_name
		        else '.'.join((self.block.name, self.name)))
