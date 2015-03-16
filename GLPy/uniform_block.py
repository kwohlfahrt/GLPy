from OpenGL import GL
from numpy import dtype

from .GLSL import ( UniformBlock, UniformBlockMember, BlockLayout,
                    Scalar, Vector, Matrix, BasicType, Struct, Array )

import ctypes as c

class ProgramUniformBlockMember(UniformBlockMember):
	def __init__(self, block, name, datatype, matrix_layout=None):
		super().__init__(block, name, datatype, matrix_layout)
	
	def __iter__(self):
		ubms = ( ProgramUniformBlockMember(ubm.block, ubm.name, ubm.datatype, ubm.shader_matrix_layout)
				 for ubm in super().__iter__() )
		yield from filter(lambda m: m.active, ubms)

	
	def __getitem__(self, idx):
		ubm = super().__getitem__(idx)
		return ProgramUniformBlockMember(ubm.block, ubm.name, ubm.datatype, ubm.shader_matrix_layout)
	
	@property
	def active(self):
		if isinstance(getattr(self.datatype, 'base', self.datatype), Struct):
			return any(m.active is not None for m in self)
		return self.index is not None

	@property
	def program(self):
		return self.block.program
	
	# Query index of one member at a time, this is the approach in GL.glGetProgramResourceIndex
	# which superseeds GL.glGetUniform* interface
	@property
	def index(self):
		'''The uniform index of the block member. Will return :py:obj:`None` if it is not a resource
		(i.e. a :py:class:`.Struct` type) , or if it is not active.

		:rtype: :py:obj:`int` or :py:obj:`None`
		'''
		if isinstance(getattr(self.datatype, 'base', self.datatype), Struct):
			return None

		names = c.pointer(c.c_char_p(self.api_name.encode()))
		names = c.cast(names, c.POINTER(c.POINTER(c.c_char)))

		out = GL.GLuint()
		GL.glGetUniformIndices(self.program.handle, 1, names, c.byref(out))
		return out.value if out.value != GL.GLuint(GL.GL_INVALID_INDEX).value else None

	@property
	def offset(self):
		'''Returns the offset of the member from the start of the parent :py:class:`UniformBlock`

		:rtype: :py:obj:`int`
		'''
		if not self.active:
			return None

		if isinstance(self.datatype, Struct):
			return min(m.offset for m in self)
		elif isinstance(self.datatype, Array):
			return self[0].offset

		offset = GL.GLint()
		GL.glGetActiveUniformsiv(self.program.handle, 1, self.index,
		                         GL.GL_UNIFORM_OFFSET, c.byref(offset))
		return offset.value
	
	@property
	def dtype(self):
		'''The numpy datatype used to represent this buffer block member in a buffer. Unlike the
		implementation on :py:class:`InterfaceBlockMember`, this does not raise an exception for
		non-standard layouts.

		:rtype: :py:class:`numpy.dtype`
		'''
		if self.layout in (BlockLayout.std140, BlockLayout.std430):
			return super().dtype

		if isinstance(self.datatype, Array) and isinstance(self.datatype.base, BasicType):
			array_stride = GL.GLint()
			array_stride = GL.glGetActiveUniformsiv(self.program.handle, 1, self.index,
													GL.GL_UNIFORM_ARRAY_STRIDE, c.byref(array_stride))
			array_stride = array_stride.value
		if isinstance(getattr(self.datatype, 'base', self.datatype), Matrix):
			matrix_stride = GL.GLint()
			matrix_stride = GL.glGetActiveUniformsiv(self.program.handle, 1, self.index,
													GL.GL_UNIFORM_MATRIX_STRIDE, c.byref(matrix_stride))
			matrix_stride = matrix_stride.value

		if isinstance(self.datatype, (Scalar, Vector)):
			return self.datatype.machine_type
		elif isinstance(self.datatype, Matrix):
			if self.matrix_layout == MatrixLayout.column_major:
				items, components = self.datatype.shape
			elif self.matrix_layout == MatrixLayout.row_major:
				components, items = self.datatype.shape
			item_dtype = Vector.fromType(self.datatype.scalar_type, components).machine_type
			item_dtype = dtype({'names': [self.datatype.name], 'formats': [item_dtype],
			                    'itemsize': matrix_stride})
			return dtype((item_dtype, items))
		elif isinstance(self.datatype, Array) and isinstance(self.datatype.base, BasicType):
			item_dtype = ProgramUniformBlockMember(self.block, self.name, self.datatype.base).dtype
			item_dtype = dtype({'names': [self.datatype.base.name], 'formats': [item_dtype],
			                    'itemsize': array_stride})
			return dtype((item_dtype, self.datatype.full_shape))
		elif ( isinstance(self.datatype, Array) and isinstance(self.datatype.base, Struct)
		     or isinstance(getattr(self.datatype, 'base', self.datatype), Struct) ):
			# Have to process each member separately, as members can be removed in optimization pass
			names, formats, offsets = zip(*((m.name, m.dtype, m.offset) for m in self))
			offsets = (o - self.offset for o in offsets)
			return dtype({'names': list(names), 'formats': list(formats), 'offsets': list(offsets)})

class ProgramUniformBlock(UniformBlock):
	member_type = ProgramUniformBlockMember

	def __init__(self, program, name, *members, instance_name='', layout='shared',
	             matrix_layout='row_major', binding=None):
		self.program = program
		self.dynamic_binding = None
		super().__init__(name, *members, instance_name=instance_name, layout=layout,
		                 matrix_layout=matrix_layout, binding=binding)
		if self.layout == BlockLayout.std430:
			raise ValueError("Uniform Blocks may not have a 'std430' layout.")

	@classmethod
	def fromUniformBlock(cls, program, block):
		return cls(program, block.name, *block.members.values(), instance_name=block.instance_name,
		           layout=block.layout, binding=block.shader_binding)
	
	def __getitem__(self, idx):
		member =  self.members[idx]
		if not member.active:
			raise KeyError("{} is not an active uniform block member.".format(idx))
		return member
	
	def __iter__(self):
		if self.layout.standardized:
			yield from super().__iter__()
		else:
			members = filter(lambda m: m.active, super().__iter__())
			yield from sorted(members, key=lambda m: m.offset)
	
	def __eq__(self, other):
		return super.__eq__(other) and self.program == other.program and self.binding == other.binding

	@property
	def dtype(self):
		if self.layout.standardized:
			return super().dtype
		else:
			names, formats, offsets = zip(*((m.name, m.dtype, m.offset) for m in self))
			nbytes = GL.glGetActiveUniformBlockiv(self.program.handle, self.index,
		                                          GL.GL_UNIFORM_BLOCK_DATA_SIZE)
			return dtype({'names': list(names), 'formats': list(formats), 'offsets': list(offsets),
			              'itemsize': nbytes})

	@property
	def index(self):
		return GL.glGetUniformBlockIndex(self.program.handle, self.name)

	@property
	def binding(self):
		if self.shader_binding is not None:
			return self.shader_binding
		return self.dynamic_binding 
	
	@binding.setter
	def binding(self, binding_index):
		if self.shader_binding is not None:
			raise TypeError("This uniform block has an explicit binding.")
		GL.glUniformBlockBinding(self.program.handle, self.index, binding_index)
		self.dynamic_binding = binding_index
