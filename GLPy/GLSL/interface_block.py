from .datatypes import Scalar, Vector, Matrix, Array, Struct, BasicType
from .variable import Variable

from util.misc import roundUp
from numpy import dtype

from enum import Enum

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

	:raises TypeError: If it is passed an opaque type as a base
	'''
	def __init__(self, block, name, datatype, matrix_layout=None):
		if isinstance(datatype, Struct):
			contents = (InterfaceBlockMember(block, c.name, c.datatype, matrix_layout)
			            for c in datatype)
			datatype = Struct(datatype.name, *contents)
		elif isinstance(datatype, Array) and isinstance(datatype.base, Struct):
			contents = (InterfaceBlockMember(block, c.name, c.datatype, matrix_layout)
			            for c in datatype.base)
			datatype = Array(Struct(datatype.base.name, *contents), datatype.full_shape)
		super().__init__(name, datatype)

		if isinstance(datatype, BasicType) and datatype.opaque is True:
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

		:rtype: :py:class:`.InterfaceBlockMember`
		'''
		return cls(block, var.name, var.datatype, matrix_layout)

	@property
	def gl_name(self):
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

		:rtype: :py:obj:`int`
		'''

		if self.layout not in (BlockLayout.std140, BlockLayout.std430):
			raise NotImplementedError("Must query non-std block layouts.")

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

		# Rule 4
		if isinstance(self.datatype, Array) and isinstance(self.datatype.base, (Scalar, Vector)):
			align_type = InterfaceBlockMember(self.block, self.name, self.datatype.base)
			return roundUp(align_type.alignment, alignment_rounding)

		# Rules 5 & 7
		if isinstance(self.datatype, Matrix):
			if self.matrix_layout == MatrixLayout.column_major:
				items, components = self.datatype.shape
			elif self.matrix_layout == MatrixLayout.row_major:
				components, items = self.datatype.shape
			align_type = Array(Vector.fromType(self.datatype.scalar_type, components), items)
			return InterfaceBlockMember(self.block, self.name, align_type).alignment

		# Rules 6 & 8
		elif isinstance(self.datatype, Array) and isinstance(self.datatype.base, Matrix):
			if self.matrix_layout == MatrixLayout.column_major:
				items, components = self.datatype.base.shape
			elif self.matrix_layout == MatrixLayout.row_major:
				components, items = self.datatype.base.shape
			dims = self.datatype.full_shape + (items,)
			align_type = Array(Vector.fromType(self.datatype.base.scalar_type, components), dims)
			return InterfaceBlockMember(self.block, self.name, align_type).alignment

		# Rule 9
		if isinstance(self.datatype, Struct):
			alignment = max(c.alignment for c in self.datatype)
			return roundUp(alignment, alignment_rounding)

		# Rule 10
		if isinstance(self.datatype, Array) and isinstance(self.datatype.base, Struct):
			return InterfaceBlockMember(self.block, self.name, self.datatype.base).alignment

	@property
	def dtype(self):
		'''The numpy datatype used to represent this block member in a buffer.

		:rtype: :py:class:`numpy.dtype`
		'''
		if self.layout not in (BlockLayout.std140, BlockLayout.std430):
			raise NotImplementedError("Must query non-std block layouts.")

		# Rule 1, 2 & 3
		if isinstance(self.datatype, Scalar) or isinstance(self.datatype, Vector):
			return self.datatype.machine_type

		# Rule 4
		if isinstance(self.datatype, Array) and isinstance(self.datatype.base, (Scalar, Vector)):
			array_stride = self.alignment
			element_dtype = self.datatype.base.machine_type
			element_alignment = roundUp(element_dtype.itemsize, array_stride)
			element_dtype = dtype({'names': [self.datatype.base.name], 'formats': [element_dtype],
			                       'itemsize': element_alignment})
			return dtype((element_dtype, self.datatype.full_shape))

		# Rules 5 & 7
		if isinstance(self.datatype, Matrix):
			if self.matrix_layout == MatrixLayout.column_major:
				items, components = self.datatype.shape
			elif self.matrix_layout == MatrixLayout.row_major:
				components, items = self.datatype.shape
			datatype = Array(Vector.fromType(self.datatype.scalar_type, components), items)
			return InterfaceBlockMember(self.block, self.name, datatype).dtype

		# Rule 6 & 8
		if isinstance(self.datatype, Array) and isinstance(self.datatype.base, Matrix):
			if self.matrix_layout == MatrixLayout.column_major:
				items, components = self.datatype.base.shape
			elif self.matrix_layout == MatrixLayout.row_major:
				components, items = self.datatype.base.shape
			dims = self.datatype.full_shape + (items,)
			datatype = Array(Vector.fromType(self.datatype.base.scalar_type, components), dims)
			return InterfaceBlockMember(self.block, self.name, datatype).dtype

		# Rule 9
		if isinstance(self.datatype, Struct):
			content_dtypes = [c.dtype for c in self.datatype]
			content_alignments = [c.alignment for c in self.datatype]
			content_names = [c.name for c in self.datatype]
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
		if isinstance(self.datatype, Array) and isinstance(self.datatype.base, Struct):
			element_member = InterfaceBlockMember(self.block, self.name, self.datatype.base)
			array_stride = element_member.alignment
			element_dtype = element_member.dtype
			element_alignment = roundUp(element_dtype.itemsize, array_stride)
			element_dtype = dtype({'names': [self.datatype.base.name], 'formats': [element_dtype],
			                       'itemsize': element_alignment})
			return dtype((element_dtype, self.datatype.full_shape))
