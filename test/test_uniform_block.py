from GLPy import Program
from GLPy.GLSL import UniformBlock, UniformBlockMember, Variable, Struct, Array

from numpy import dtype

from .test_context import ContextTest, readShaders

class UniformBlockTest(ContextTest):
	def setUp(self):
		super().setUp()
		shaders = {'vertex': 'uniform_block.vert',
		           'fragment': 'compile.frag'}
		shaders = readShaders(**shaders)
		struct_foo = Struct('Foo', Variable('f', 'float'), Variable('v4', 'vec4'))
		# This has an incorrect definition to simulate members being optimized out
		struct_foo_optimized = Struct('Foo', Variable('f', 'float'), Variable('v4', 'vec4'),
		                              Variable('optimized_out', 'float'))
		uniform_blocks = [UniformBlock('AnonymousUB', Variable('m4', 'mat4'), Variable('b', 'bool'),
		                               binding=2, layout='std140'),
		                  UniformBlock('InstancedUB', Variable('f', 'float'), Variable('v4', 'vec4'),
						               Variable('foo', struct_foo), instance_name='ubinstance'),
						  # This has an incorrect definition to simulate members being optimized out
						  UniformBlock('OptimizedUB', Variable('ui', 'uint'), Variable('i', 'int'),
						               Variable('optimized_out', 'float'),
									   Variable('foo', Array(struct_foo_optimized, 2)),
									   instance_name='uboptimized')]
		self.program = Program(shaders, uniform_blocks=uniform_blocks)
	
	def test_index(self):
		# No way to determine indices in advance, so just make sure they are unique
		block_indices = {ub.index for ub in self.program.uniform_blocks.values()}
		self.assertEqual(len(block_indices), len(self.program.uniform_blocks))
	
	def test_member_indices(self):
		self.assertIsNotNone(self.program.uniform_blocks['OptimizedUB']['ui'].index)
		with self.assertRaises(KeyError):
			self.program.uniform_blocks['OptimizedUB']['optimized_out']
	
	def test_binding(self):
		self.assertEqual(self.program.uniform_blocks['AnonymousUB'].binding, 2)
		with self.assertRaises(TypeError):
			self.program.uniform_blocks['AnonymousUB'].binding = 3
		self.program.uniform_blocks['InstancedUB'].binding = 4
		self.assertEqual(self.program.uniform_blocks['InstancedUB'].binding, 4)
	
	def test_dtype(self):
		mat4_dtype = dtype((dtype([('mat4-column', 'float32', 4)]), 4))
		anonymous_ub_dtype = dtype([('m4', mat4_dtype), ('b', 'uint32')])
		self.assertEqual(self.program.uniform_blocks['AnonymousUB'].dtype, anonymous_ub_dtype)
		packed_dtype = self.program.uniform_blocks['InstancedUB'].dtype
		ubo = self.program.uniform_blocks['OptimizedUB']
		optimized_dtype = self.program.uniform_blocks['OptimizedUB'].dtype
