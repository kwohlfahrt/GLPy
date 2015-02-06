import unittest
from GLPy import Scalar, Vector, Matrix, Variable, Struct, Array
from GLPy import Sampler

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
