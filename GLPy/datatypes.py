from OpenGL import GL
from OpenGL.constants import GLboolean,GLint, GLuint, GLfloat, GLdouble
from itertools import repeat, chain, count, product as cartesian
from collections import OrderedDict

from numpy import dtype

from util.misc import product, roundUp
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

class BlockLayout(Enum):
	shared = 1
	packed = 2
	std140 = 3
	std430 = 4

class MatrixLayout(Enum):
	column_major = 1
	row_major = 2

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

	def __init__(self, name, *members, instance_name='',
	             layout=BlockLayout.packed, matrix_layout=MatrixLayout.column_major):
		self.name = name
		self.members = {m.name: InterfaceBlockMember.fromVariable(self, m) for m in members}
		self.instance_name = instance_name
		self.layout = layout
		self.matrix_layout = matrix_layout

	@property
	def dtype(self):
		if self.layout == BlockMemoryLayout.std140:
			raise NotImplementedError("TODO")
		elif self.layout == BlockMemoryLayout.std430:
			raise NotImplementedError("TODO")
		else:
			raise TypeError("The layout for this interface block is not defined.")

	def __iter__(self):
		yield from self.members

class InterfaceBlockMember(Variable):
	'''A variable that is a member of an interface block.

	Constructed implicitly from contents passed to a :py:class:`.InterfaceBlock`.

	:param block: The block this member belongs to.
	:type block: :py:class:`.InterfaceBlockMember`
	:param matrix_layout: The layout for this member if it is a matrix, or the default layout for
	  any matrices in this member if it is a struct.
	:type MatrixLayout: :py:class:`.MatrixLayout`

	:raises TypeError: If it is passed an opaque type as a base'''
	def __init__(self, block, name, gl_type, matrix_layout=None):
		if isinstance(gl_type, Struct):
			contents = (InterfaceBlockMember(block, c.name, c.type, matrix_layout) for c in gl_type)
			gl_type = Struct(gl_type.name, *contents)
		elif isinstance(gl_type, Array) and isinstance(gl_type.base, Struct):
			contents = (InterfaceBlockMember(block, c.name, c.type, matrix_layout) for c in gl_type.base)
			gl_type = Array(Struct(gl_type.base.name, *contents), gl_type.full_shape)
		super().__init__(name, gl_type)

		if isinstance(gl_type, BasicType) and gl_type.opaque is True:
			raise TypeError("Interface blocks may not contain opaque types.")
		self._matrix_layout = matrix_layout
		self.block = block

	@classmethod
	def fromVariable(cls, block, var, matrix_layout=None):
		'''Construct from a block and a :py:class:`.Variable`

		:param block: The block the variable belongs to
		:type block: :py:class:`.InterfaceBlock`
		:param var: The variable describing the block member
		:type var: :py:class:`.Variable`
		:param matrix_layout: The matrix layout of this member, or :py:obj:`None` if it is to be
		  inherited from the parent.
		:type matrix_layout: :py:class:`MatrixLayout` or :py:obj:`None`
		'''
		return cls(block, var.name, var.type, matrix_layout)

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

	@property
	def matrix_layout(self):
		if self._matrix_layout is None:
			return self.block.matrix_layout
		return self._matrix_layout

	@property
	def layout(self):
		return self.block.layout

	@property
	def alignment(self):
		'''How the block member is to be aligned. This is only defined if the block layout is
		:py:obj:`BlockLayout.std140` or :py:obj:`BlockLayout.std430`

		:returns: The alignment of the block member.
		:rtype: :py:obj:`int`
		'''

		if self.layout not in (BlockLayout.std140, BlockLayout.std430):
			raise NotImplementedError("Must query non-std block layouts.")

		if self.layout == BlockLayout.std140:
			alignment_rounding = Vector.vec4.machine_type.itemsize
		else:
			alignment_rounding = 1

		# Rule 1
		if isinstance(self.type, Scalar):
			return self.type.machine_type.itemsize
		# Rule 2 & 3
		elif isinstance(self.type, Vector):
			item_type, (components,) = self.type.machine_type.subdtype
			if components == 3:
				components = 4
			return item_type.itemsize * components

		# Rule 4
		if isinstance(self.type, Array) and isinstance(self.type.base, (Scalar, Vector)):
			align_type = InterfaceBlockMember(self.block, self.name, self.type.base)
			return roundUp(align_type.alignment, alignment_rounding)

		# Rules 5 & 7
		if isinstance(self.type, Matrix):
			if self.matrix_layout == MatrixLayout.column_major:
				items, components = self.type.shape
			elif self.matrix_layout == MatrixLayout.row_major:
				components, items = self.type.shape
			align_type = Array(Vector.fromType(self.type.scalar_type, components), items)
			return InterfaceBlockMember(self.block, self.name, align_type).alignment

		# Rules 6 & 8
		elif isinstance(self.type, Array) and isinstance(self.type.base, Matrix):
			if self.matrix_layout == MatrixLayout.column_major:
				items, components = self.type.base.shape
			elif self.matrix_layout == MatrixLayout.row_major:
				components, items = self.type.base.shape
			dims = self.type.full_shape + (items,)
			align_type = Array(Vector.fromType(self.type.base.scalar_type, components), dims)
			return InterfaceBlockMember(self.block, self.name, align_type).alignment

		# Rule 9
		if isinstance(self.type, Struct):
			alignment = max(c.alignment for c in self.type)
			return roundUp(alignment, alignment_rounding)

		# Rule 10
		if isinstance(self.type, Array) and isinstance(self.type.base, Struct):
			return InterfaceBlockMember(self.block, self.name, self.type.base).alignment

	@property
	def dtype(self):
		if self.layout not in (BlockLayout.std140, BlockLayout.std430):
			raise NotImplementedError("Must query non-std block layouts.")

		# Rule 1, 2 & 3
		if isinstance(self.type, Scalar) or isinstance(self.type, Vector):
			return self.type.machine_type

		# Rule 4
		if isinstance(self.type, Array) and isinstance(self.type.base, (Scalar, Vector)):
			array_stride = self.alignment
			element_dtype = self.type.base.machine_type
			element_alignment = roundUp(element_dtype.itemsize, array_stride)
			element_dtype = dtype({'names': [self.type.base.name], 'formats': [element_dtype],
			                       'itemsize': element_alignment})
			return dtype((element_dtype, self.type.full_shape))

		# Rules 5 & 7
		if isinstance(self.type, Matrix):
			if self.matrix_layout == MatrixLayout.column_major:
				items, components = self.type.shape
			elif self.matrix_layout == MatrixLayout.row_major:
				components, items = self.type.shape
			gl_type = Array(Vector.fromType(self.type.scalar_type, components), items)
			return InterfaceBlockMember(self.block, self.name, gl_type).dtype

		# Rule 6 & 8
		if isinstance(self.type, Array) and isinstance(self.type.base, Matrix):
			if self.matrix_layout == MatrixLayout.column_major:
				items, components = self.type.base.shape
			elif self.matrix_layout == MatrixLayout.row_major:
				components, items = self.type.base.shape
			dims = self.type.full_shape + (items,)
			gl_type = Array(Vector.fromType(self.type.base.scalar_type, components), dims)
			return InterfaceBlockMember(self.block, self.name, gl_type).dtype

		# Rule 9
		if isinstance(self.type, Struct):
			content_dtypes = [c.dtype for c in self.type]
			content_alignments = [c.alignment for c in self.type]
			content_names = [c.name for c in self.type]
			content_offsets = []
			offset = 0
			for dt, alignment in zip(content_dtypes, content_alignments):
				offset = roundUp(offset, alignment)
				content_offsets.append(offset)
				offset += dt.itemsize
			struct_size = roundUp(offset, self.alignment)
			return dtype({'names': content_names, 'formats': content_dtypes,
			              'offsets': content_offsets, 'itemsize': struct_size})

		# Rule 10
		if isinstance(self.type, Array) and isinstance(self.type.base, Struct):
			element_member = InterfaceBlockMember(self.block, self.name, self.type.base)
			array_stride = element_member.alignment
			element_dtype = element_member.dtype
			element_alignment = roundUp(element_dtype.itemsize, array_stride)
			element_dtype = dtype({'names': [self.type.base.name], 'formats': [element_dtype],
			                       'itemsize': element_alignment})
			return dtype((element_dtype, self.type.full_shape))
