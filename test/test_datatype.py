import unittest
from GLPy import Scalar, Vector, Matrix, Variable, Struct, Array
from GLPy.datatypes import InterfaceBlock, InterfaceBlockMember
from GLPy import Sampler
from GLPy import BlockLayout, MatrixLayout
from numpy import dtype

from itertools import chain, product as cartesian

class TestVariable(unittest.TestCase):
	def test_from_str(self):
		v = Variable('foo', 'vec3')
		self.assertEqual(v, Variable('foo', Vector.vec3))

	def test_str(self):
		v = Variable('foo', 'vec3')
		self.assertEqual(str(v), 'vec3 foo')

		v = Variable('foo', Array('vec3', 3))
		self.assertEqual(str(v), 'vec3[3] foo')

	def test_struct(self):
		a = Variable('first', 'int')
		b = Variable('second', 'vec3')
		v = Variable('bar', Struct('TheStruct', a, b))

		self.assertEqual(str(v), "TheStruct bar")
		self.assertEqual(len(v), 2)
		self.assertEqual(next(iter(v)), Variable('bar.first', 'int'))
		self.assertEqual(v['first'], Variable('bar.first', 'int'))
		with self.assertRaises(KeyError):
			v['foo']
		with self.assertRaises(KeyError):
			v[1]

	def test_resources(self):
		v = Variable('a', 'int')
		self.assertEqual(v.resources, [v])

		basic_struct = Struct('StructOfBasics', Variable('first', 'int'), Variable('second', 'vec3'))
		struct_var = Variable('foo', basic_struct)
		self.assertEqual(struct_var.resources, [ Variable('foo.first', 'int')
		                                       , Variable('foo.second', 'vec3')])

		array_var = Variable('bar', Array('vec3', 3))
		self.assertEqual(array_var.resources, [Variable('bar[0]', Array('vec3', 3))])

		array_array_var = Variable('baz', Array('mat4', (5, 6)))
		self.assertEqual(array_array_var.resources, [Variable('baz[{}][0]'.format(i), Array('mat4', 6))
		                                             for i in range(5)])

		array_struct = Struct('StructOfArrays', Variable('first', Array('vec3', 5))
		                                      , Variable('second', 'mat2')
		                                      , Variable('third', Array('int', (3, 4))))

		array_struct_var = Variable('foobar', array_struct)
		as_resources = ( [ Variable('foobar.first[0]', Array('vec3', 5))
		                 , Variable('foobar.second', 'mat2')]
					   + [ Variable('foobar.third[{}][0]'.format(i), Array('int', 4))
					       for i in range(3) ] )
		self.assertEqual(array_struct_var.resources, as_resources)

		struct_array_var = Variable('barfoo', Array(basic_struct, (5, 6)))
		sa_resources = [ (Variable('barfoo[{}][{}].first'.format(i, j), 'int')
		                 ,Variable('barfoo[{}][{}].second'.format(i, j), 'vec3'))
						 for i, j in cartesian(range(5), range(6)) ]
		sa_resources = list(chain.from_iterable(sa_resources))
		self.assertEqual(struct_array_var.resources, sa_resources)

class TestArray(unittest.TestCase):
	def test_simple(self):
		a = Array('vec3', 3)
		self.assertEqual(len(a), 3)
		self.assertEqual(a.element, 'vec3')
		self.assertEqual(list(iter(a)), [Vector.vec3] * 3)
		self.assertEqual(a[0], 'vec3')
		self.assertEqual(a.full_shape, (3,))
		self.assertEqual(a.base, 'vec3')
		self.assertEqual(str(a), 'vec3[3]')

	def test_multidim(self):
		a = Array('vec3', (3, 4))
		self.assertEqual(len(a), 3)
		self.assertEqual(a.element, Array('vec3', 4))
		self.assertEqual(list(iter(a)), [Array('vec3', 4)] * 3)
		self.assertEqual(a[0], Array('vec3', 4))
		self.assertEqual(a.full_shape, (3, 4))
		self.assertEqual(a.base, 'vec3')
		self.assertEqual(str(a), 'vec3[3][4]')

	def test_struct_array(self):
		a = Array(Struct('Foo', Variable('first', 'vec3'), Variable('second', 'int')), 3)
		self.assertEqual(str(a), 'Foo[3]')

		a = Array(Struct('Foo', Variable('first', 'vec3'), Variable('second', 'int')), (3, 4))
		self.assertEqual(str(a), 'Foo[3][4]')

class TestStruct(unittest.TestCase):
	def test_getitem(self):
		s = Struct( 'TheStruct', Variable('first', 'vec3')
		          ,  Variable('second', Array('float', (4, 5))))
		self.assertEqual(s['first'], Variable('first', 'vec3'))
		self.assertEqual(s['second'], Variable('second', Array('float', (4, 5))))

class TestBasicType(unittest.TestCase):
	def test_machine_type(self):
		self.assertEqual(Vector('vec3').machine_type, dtype(('float32', 3)))

class TestInterfaceBlock(unittest.TestCase):
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
		expected = dtype({'names': ['v3', 'f'], 'offsets': [0, 12],
		                  'formats': [dtype(('float32', 3)), dtype('float32')],
		                  'itemsize': 16})
		self.assertEqual(i.dtype, expected)
		struct = Struct('Bar', Variable('f', 'float'), Variable('v3', 'vec3'))
		i = InterfaceBlockMember(block, 'foo', struct)
		expected = dtype({'names': ['f', 'v3'], 'offsets': [0, 16],
		                  'formats': [dtype('float32'), dtype(('float32', 3))],
		                  'itemsize': 32})
		self.assertEqual(i.dtype, expected)

	def test_struct_array(self):
		block = InterfaceBlock('Foo', layout=BlockLayout.std140)
		struct = Struct('Bar', Variable('f', 'float'), Variable('v3', 'vec3'))
		i = InterfaceBlockMember(block, 'foo', Array(struct, 4))
		struct_dtype = dtype({'names': ['f', 'v3'], 'offsets': [0, 16],
		                      'formats': [dtype('float32'), dtype(('float32', 3))],
		                      'itemsize': 32})
		expected = dtype(([('Bar', struct_dtype)], 4))
		self.assertEqual(i.dtype, expected)
