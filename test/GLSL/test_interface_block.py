import unittest
from numpy import dtype

from itertools import product as cartesian

from GLPy.GLSL.interface_block import *

class TestInterfaceBlock(unittest.TestCase):
	def test_naming(self):
		baz = Struct('Baz', Variable('f', 'float'), Variable('v3', 'vec3'))
		block_members = [ Variable('b', baz), Variable('f', 'float'), Variable('bs', Array(baz, 3)),
		                  Variable('m4s', Array('mat4', 3))]
		block = InterfaceBlock('IBlock', *block_members, layout='std140')
		self.assertEqual(block['b'].api_name, 'b')
		self.assertEqual(block['b'].glsl_name, 'b')
		block = InterfaceBlock('IBlock', *block_members, instance_name='blockiname', layout='std140')
		self.assertEqual(block['b'].api_name, 'IBlock.b')
		self.assertEqual(block['b'].glsl_name, 'blockiname.b')

	def test_opaque_error(self):
		struct = Struct('Baz', Variable('s2d', 'sampler2D'), Variable('v3', 'vec3'))
		with self.assertRaises(TypeError):
			InterfaceBlock('IBlock', Variable('baz', struct), Variable('f', 'float'))
		with self.assertRaises(TypeError):
			InterfaceBlock('IBlock', Variable('s2d', 'sampler2D'), Variable('f', 'float'))

class TestInterfaceBlockMember(unittest.TestCase):
	def test_from_var(self):
		block = InterfaceBlock('Foo')
		v = Variable('f', 'float')
		self.assertEqual(InterfaceBlockMember.fromVariable(block, v),
		                 InterfaceBlockMember(block, 'f', 'float'))
		self.assertEqual(InterfaceBlockMember.fromVariable(block, v, 'row_major'),
		                 InterfaceBlockMember(block, 'f', 'float', 'row_major'))
		self.assertNotEqual(InterfaceBlockMember.fromVariable(block, v, 'row_major'),
		                 InterfaceBlockMember(block, 'f', 'float'))

	def test_opaque_error(self):
		block = InterfaceBlock('Foo')
		with self.assertRaises(TypeError):
			InterfaceBlockMember(block, 'foo', 'sampler2D')
		struct = Struct('Baz', Variable('s2d', 'sampler2D'), Variable('v3', 'vec3'))
		with self.assertRaises(TypeError):
			InterfaceBlockMember(block, 'foo', struct)

	def test_vector_alignment(self):
		block = InterfaceBlock('Foo', layout=BlockLayout.std140)
		i = InterfaceBlockMember(block, 'foo', 'float')
		self.assertEqual(i.alignment, 4)
		i = InterfaceBlockMember(block, 'foo', 'vec2')
		self.assertEqual(i.alignment, 8)
		i = InterfaceBlockMember(block, 'foo', 'vec3')
		self.assertEqual(i.alignment, 16)
		i = InterfaceBlockMember(block, 'foo', 'vec4')
		self.assertEqual(i.alignment, 16)

	def test_matrix_alignment(self):
		block = InterfaceBlock('Foo', layout=BlockLayout.std140)
		for r, c in cartesian(range(2, 5), repeat=2):
			mat = 'mat{}x{}'.format(r, c)
			i = InterfaceBlockMember(block, 'foo', mat)
			self.assertEqual(i.alignment, 16)

	def test_array_alignment(self):
		block = InterfaceBlock('Foo', layout=BlockLayout.std140)
		for t in ['float', 'vec2', 'vec3', 'vec4', 'mat2', 'mat4']:
			i = InterfaceBlockMember(block, 'foo', Array(t, 2))
			self.assertEqual(i.alignment, 16)

	def test_struct_alignment(self):
		block = InterfaceBlock('Foo', layout=BlockLayout.std140)
		struct = Struct('Bar', Variable('v3', 'vec3'), Variable('f', 'float'))
		i = InterfaceBlockMember(block, 'bar', struct)
		self.assertEqual(i.alignment, 16)

		nested_struct = Struct('Baz', Variable('s', struct), Variable('f', 'float'))
		i = InterfaceBlockMember(block, 'baz', nested_struct)
		self.assertEqual(i.alignment, 16)

	def test_struct_array_alignment(self):
		block = InterfaceBlock('Foo', layout=BlockLayout.std140)
		struct = Struct('Bar', Variable('v3', 'vec3'), Variable('f', 'float'))
		i = InterfaceBlockMember(block, 'bar', Array(struct, 4))
		self.assertEqual(i.alignment, 16)

	def test_vector_dtype(self):
		block = InterfaceBlock('Foo', layout=BlockLayout.std140)
		i = InterfaceBlockMember(block, 'foo', 'float')
		self.assertEqual(i.dtype, dtype('float32'))
		i = InterfaceBlockMember(block, 'foo', 'ivec2')
		self.assertEqual(i.dtype, dtype(('int32', 2)))
		i = InterfaceBlockMember(block, 'foo', 'bvec3')
		self.assertEqual(i.dtype, dtype(('uint32', 3)))
		i = InterfaceBlockMember(block, 'foo', 'uvec4')
		self.assertEqual(i.dtype, dtype(('uint32', 4)))

	def test_matrix_dtype(self):
		block = InterfaceBlock('Foo', layout=BlockLayout.std140)
		i = InterfaceBlockMember(block, 'foo', 'mat2x3')
		element_dtype = dtype({'names': ['vec3'], 'formats': [dtype(('float32', (3)))], 'itemsize': 16})
		expected = dtype((element_dtype, (2,)))
		self.assertEqual(i.dtype, expected)

	def test_array_dtype(self):
		block = InterfaceBlock('Foo', layout=BlockLayout.std140)
		i = InterfaceBlockMember(block, 'foo', Array('mat2x3', 4))
		element_dtype = dtype({'names': ['vec3'], 'formats': [dtype(('float32', (3)))], 'itemsize': 16})
		expected = dtype((element_dtype, (4, 2)))
		self.assertEqual(i.dtype, expected)
		i = InterfaceBlockMember(block, 'foo', Array('int', 4))
		element_dtype = dtype({'names': ['int'], 'formats': [dtype('int32')], 'itemsize': 16})
		expected = dtype((element_dtype, (4,)))
		self.assertEqual(i.dtype, expected)

	def test_struct_dtype(self):
		block = InterfaceBlock('Foo', layout=BlockLayout.std140)
		struct = Struct('Bar', Variable('v3', 'vec3'), Variable('f', 'float'))
		i = InterfaceBlockMember(block, 'foo', struct)
		expected = dtype({'names': ['foo.v3', 'foo.f'], 'offsets': [0, 12],
		                  'formats': [dtype(('float32', 3)), dtype('float32')],
		                  'itemsize': 16})
		self.assertEqual(i.dtype, expected)
		struct = Struct('Bar', Variable('f', 'float'), Variable('v3', 'vec3'))
		i = InterfaceBlockMember(block, 'foo', struct)
		expected = dtype({'names': ['foo.f', 'foo.v3'], 'offsets': [0, 16],
		                  'formats': [dtype('float32'), dtype(('float32', 3))],
		                  'itemsize': 32})
		self.assertEqual(i.dtype, expected)

	def test_struct_array(self):
		block = InterfaceBlock('Foo', layout=BlockLayout.std140)
		struct = Struct('Bar', Variable('f', 'float'), Variable('v3', 'vec3'))
		i = InterfaceBlockMember(block, 'foo', Array(struct, 4))
		struct_dtype = dtype({'names': ['foo.f', 'foo.v3'], 'offsets': [0, 16],
		                      'formats': [dtype('float32'), dtype(('float32', 3))],
		                      'itemsize': 32})
		expected = dtype(([('Bar', struct_dtype)], 4))
		self.assertEqual(i.dtype, expected)
