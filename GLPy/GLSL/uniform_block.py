from .interface_block import InterfaceBlock, InterfaceBlockMember, BlockLayout, MatrixLayout

class UniformBlockMember(InterfaceBlockMember):
	# TODO: Explicit layout (ARB_enhanced_layout)
	@classmethod
	def fromInterfaceBlockMember(cls, member):
		return cls(member.block, member.name, member.datatype, member.shader_matrix_layout)

	def __iter__(self):
		yield from (UniformBlockMember.fromInterfaceBlockMember(m) for m in super().__iter__())
	
	def __getitem__(self, idx):
		return UniformBlockMember.fromInterfaceBlockMember(super().__getitem__(idx))

class UniformBlock(InterfaceBlock):
	'''An OpenGL Uniform Block.'''

	member_type = UniformBlockMember
	def __init__(self, name, *members, instance_name='', layout='shared', matrix_layout='row_major',
	             binding=None):
		super().__init__(name, *members, instance_name=instance_name, layout=layout)
		self.shader_binding = binding

	def __str__(self):
		members = '; '.join(str(m) for m in self)
		return "uniform {} {{{};}} {};".format(self.name, members, self.instance_name)
