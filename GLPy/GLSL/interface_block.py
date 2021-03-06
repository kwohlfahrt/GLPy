from .datatypes import Scalar, Vector, Matrix, Array, Struct, BasicType
from .variable import Variable

from util.misc import roundUp
from numpy import dtype

from enum import Enum
from collections import OrderedDict

class BlockLayout(Enum):
	'''The valid layouts for interface blocks.

	Defines the following attributes:

	*standardized*
	  Whether the memory layout of a a block of this layout can be determined in advance (i.e.
	  without runtime queries)
	'''
	def __init__(self, value):
		self.standardized = self.name.startswith('std')

	shared = 'shared'
	packed = 'packed'
	std140 = 'std140'
	std430 = 'std430'

class MatrixLayout(Enum):
	'''The valid options for matrix layouts in interface blocks and interface block members.'''

	column_major = 'column_major'
	row_major = 'row_major'

class InterfaceBlockMember(Variable):
	'''A variable that is a member of an interface block.

	:param block: The block this member belongs to.
	:type block: :py:class:`.InterfaceBlockMember`
	:param matrix_layout: The layout for this member if it is a matrix, or the default layout for
	  any matrices in this member if it is a struct.
	:type matrix_layout: :py:class:`.MatrixLayout` or :py:obj:`None`

	:raises TypeError: If it is passed an opaque type as a base
	'''
	def __init__(self, block, name, datatype, matrix_layout=None):
		super().__init__(name, datatype)
		self.shader_matrix_layout = matrix_layout
		self.block = block
		if any(getattr(r.datatype, 'base', r.datatype).opaque for r in self.resources):
			raise TypeError("Interface blocks may not contain opaque types.")

	@classmethod
	def fromVariable(cls, block, var, matrix_layout=None):
		'''Construct from a block and a :py:class:`.Variable`.

		:param block: The block the variable belongs to
		:type block: :py:class:`.InterfaceBlock`
		:param var: The variable describing the block member
		:type var: :py:class:`.Variable`
		:param matrix_layout: The matrix layout of this member, or :py:obj:`None` if it is to be
		  inherited from the parent.
		:type matrix_layout: :py:class:`MatrixLayout` or :py:obj:`None`

		:rtype: :py:class:`.InterfaceBlockMember`
		'''
		return cls(block, var.name, var.datatype, matrix_layout)

	def __getitem__(self, idx):
		var = super().__getitem__(idx)
		return InterfaceBlockMember.fromVariable(self.block, var, self.shader_matrix_layout)

	def __iter__(self):
		''':rtype: [:py:class:`InterfaceBlockMember`]'''
		yield from (InterfaceBlockMember.fromVariable(self.block, var, self.shader_matrix_layout)
		            for var in super().__iter__())

	def __eq__(self, other):
		return ( super().__eq__(other) and self.block == other.block
		       and self.matrix_layout == other.matrix_layout )

	@property
	def matrix_layout(self):
		''':rtype: :py:class:`MatrixLayout`'''
		if self.shader_matrix_layout is None:
			return self.block.matrix_layout
		return self.shader_matrix_layout

	@property
	def glsl_name(self):
		'''The string used to refer to the block member in a shader

		:rtype: :py:obj:`str`
		'''
		return (self.name if not self.block.instance_name
		        else '.'.join((self.block.instance_name, self.name)))

	@property
	def api_name(self):
		'''The string used to refer to the block member in the OpenGL API

		:rtype: :py:obj:`str`
		'''
		return (self.name if not self.block.instance_name
		        else '.'.join((self.block.name, self.name)))

	@property
	def matrix_stride(self):
		if not (isinstance(self.datatype, Matrix) and self.layout.standardized):
			raise TypeError("Not a matrix with standardized layout.")
		# FIXME: doubles?
		return self.alignment

	@property
	def array_stride(self):
		if not (isinstance(self.datatype, Array) and self.layout.standardized):
			raise TypeError("Not an array with standardized layout.")
		return roundUp(self[0].dtype.itemsize, self.alignment)

	@property
	def layout(self):
		return self.block.layout

	@property
	def alignment(self):
		'''How the block member is to be aligned. Returns :py:obj:`None` if the layout of the block
		containing this object is not a standardized layout.

		:rtype: :py:obj:`int` or :py:obj:`None`
		'''

		if self.layout not in (BlockLayout.std140, BlockLayout.std430):
			return None

		if self.layout == BlockLayout.std140:
			alignment_rounding = Vector.vec4.machine_type.itemsize
		else:
			alignment_rounding = 1

		# Rule 1
		if isinstance(self.datatype, Scalar):
			return self.datatype.machine_type.itemsize
		# Rule 2 & 3
		elif isinstance(self.datatype, Vector):
			item_type, (components,) = self.datatype.machine_type.subdtype
			if components == 3:
				components = 4
			return item_type.itemsize * components

		# Rule 4, 6, 7 & 10
		if isinstance(self.datatype, Array):
			return roundUp(self[0].alignment, alignment_rounding)

		# Rules 5 & 7
		if isinstance(self.datatype, Matrix):
			if self.matrix_layout == MatrixLayout.column_major:
				items, components = self.datatype.shape
			elif self.matrix_layout == MatrixLayout.row_major:
				components, items = self.datatype.shape
			align_type = Array(Vector.fromType(self.datatype.scalar_type, components), items)
			return InterfaceBlockMember(self.block, self.name, align_type).alignment

		# Rule 9
		if isinstance(self.datatype, Struct):
			alignment = max(c.alignment for c in self)
			return roundUp(alignment, alignment_rounding)

	@property
	def dtype(self):
		'''The numpy datatype used to represent this block member in a buffer.

		:rtype: :py:class:`numpy.dtype`
		:raises TypeError: If the block layout is not a standard layout.
		'''

		# Rule 1, 2 & 3
		if isinstance(self.datatype, (Scalar, Vector)):
			return self.datatype.machine_type

		# Rule 10 could be defined separately to be consisted with packed/shared layout dtypes.
		# However, all members are guaranteed to be preserved in standard formats, and this produces
		# dtypes that are easier to work with

		# Rule 4, 6, 8 & 10
		if isinstance(self.datatype, Array):
			element = self[0]
			item_dtype = element.dtype
			if isinstance(element.datatype, BasicType):
				item_dtype = dtype({'names': [element.datatype.name], 'formats': [item_dtype],
									'itemsize': self.array_stride})
			return dtype((item_dtype, len(self)))

		# Rules 5 & 7
		if isinstance(self.datatype, Matrix):
			if self.matrix_layout == MatrixLayout.column_major:
				item_dim = 'column'
				items, components = self.datatype.shape
			elif self.matrix_layout == MatrixLayout.row_major:
				item_dim = 'row'
				components, items = self.datatype.shape
			item_dtype = Vector.fromType(self.datatype.scalar_type, components).machine_type
			item_dtype = dtype({'names': ['-'.join((self.datatype.name, item_dim))],
			                    'formats': [item_dtype], 'itemsize': self.matrix_stride})
			return dtype((item_dtype, items))

		# Rule 9
		if isinstance(self.datatype, Struct):
			names = [c.name for c in self]
			dtypes = [c.dtype for c in self]
			offsets = []
			offset = 0
			for c in self:
				offset = roundUp(offset, c.alignment)
				offsets.append(offset)
				offset += c.dtype.itemsize
			struct_size = roundUp(offset, self.alignment)
			return dtype({'names': names, 'formats': dtypes, 'offsets': offsets,
			              'itemsize': struct_size})

class InterfaceBlock:
	'''A generic interface block.

	Not to be instantiated directly, but as a base for defined block types.

	:param str name: The name of the uniform block
	:param \\*members: The members of the uniform block. They may not contain
	  opaque types (e.g. :py:class:`.Sampler`)
	:type \\*members: :py:class:`.Variable`
	:param str instance_name: The name of the instance
	:param layout: The layout of the interface block
	:type layout: :py:class:`BlockLayout`
	:param matrix_layout: The matrix layout of the interface block
	:type matrix_layout: :py:class:`MatrixLayout`
	:raises ValueError: If an instance name is not defined and the block has a shape
	  larger than (1,)
	:raises: See :py:class:`InterfaceBlockMember` for additional exceptions that might be raised.
	'''

	member_type = InterfaceBlockMember

	def __init__(self, name, *members, instance_name='', layout='packed',
	             matrix_layout='column_major'):
		self.name = name
		self.instance_name = instance_name
		self.layout = BlockLayout(layout)
		self.matrix_layout = MatrixLayout(matrix_layout)

		members = ((m.name, self.member_type.fromVariable(self, m)) for m in members)
		if self.layout.standardized:
			self.members = OrderedDict(members)
		else:
			self.members = dict(members)

	@property
	def dtype(self):
		'''The memory layout of the uniform block.

		:rtype: :py:class:`numpy.dtype`
		:raises TypeError: If the block layout is not a standard layout.
		'''

		if self.layout.standardized:
			offsets = []
			offset = 0
			for m in self:
				offset = roundUp(offset, m.alignment)
				offsets.append(offset)
				offset += m.dtype.itemsize
			names, formats = zip(*((m.name, m.dtype) for m in self))
			return dtype({'names': list(names), 'formats': list(formats), 'offsets': offsets})
		else:
			raise TypeError("The layout for this interface block is not defined.")

	def __iter__(self):
		yield from self.members.values()

	def __getitem__(self, idx):
		return self.members[idx]
