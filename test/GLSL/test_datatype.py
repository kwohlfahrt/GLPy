import unittest
from GLPy.GLSL import Scalar, Vector, Matrix, Variable, Struct, Array, Sampler
from numpy import dtype

from itertools import product as cartesian

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

class TestVector(unittest.TestCase):
	def test_machine_type(self):
		self.assertEqual(Vector('vec3').machine_type, dtype(('float32', 3)))

class TestArray(unittest.TestCase):
	def test_indexing(self):
		self.assertEqual(Matrix('mat3x2')[2], Vector('vec2'))
		with self.assertRaises(IndexError):
			Matrix('mat3x2')[3]
