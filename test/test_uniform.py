from OpenGL import GL

import numpy
from numpy import dtype
from numpy import testing as np_test

import unittest

from .test_context import ContextTest, readShaders

from GLPy import Program, Uniform, UniformBlock, Variable, Struct, Array

class UniformTest(ContextTest):
	def setUp(self):
		super().setUp()

		shader_files = { 'vertex': 'uniform.vert'
		               , 'fragment': 'compile.frag' }
		shaders = readShaders(**shader_files)

		TheStruct = Struct("TheStruct", Variable('first', 'vec3')
		                  , Variable('second', 'vec4'), Variable('third', 'mat4x3'))

		uniforms = [ Uniform('aUniform', 'vec3')
		           , Uniform('aScalarUniform', 'int')
		           , Uniform('aStructUniform', TheStruct)
		           , Uniform('matrixArrayUniform', Array('mat4', 2))
				   # GL_ARB_explicit_uniform_location not in mesa
		           # , Uniform('uniformArrayOfStructs', Array(TheStruct, 2), location=1) ]
		           , Uniform('uniformArrayOfStructs', Array(TheStruct, 2)) ]
		self.program = Program(shaders, uniforms=uniforms)

	def tearDown(self):
		super().tearDown()

	def test_str(self):
		self.assertEqual(str(self.program.uniforms['aUniform']), 'uniform vec3 aUniform')
	
	def test_resources(self):
		u = self.program.uniforms
		self.assertEqual(u['aUniform'], Uniform('aUniform', 'vec3'))
		self.assertEqual(u['aStructUniform.first'], Uniform('aStructUniform.first', 'vec3'))
		self.assertEqual(u['aStructUniform.third'], Uniform('aStructUniform.third', 'mat4x3'))
		self.assertEqual(u['matrixArrayUniform[0]'],
		                 Uniform('matrixArrayUniform[0]', Array('mat4', 2)))
		self.assertEqual(u['aStructUniform.third'], Uniform('aStructUniform.third', 'mat4x3'))
		self.assertEqual(u['uniformArrayOfStructs[0].first'],
		                 Uniform('uniformArrayOfStructs[0].first', 'vec3'))
		self.assertEqual(u['uniformArrayOfStructs[1].first'],
		                 Uniform('uniformArrayOfStructs[1].first', 'vec3'))
	
	@unittest.skip("GL_ARB_explicit_uniform_location not in mesa.")
	def test_explicit_location(self):
		u = self.program.uniforms
		self.assertEqual(u['uniformArrayOfStructs[0].first'].location, 1)
		self.assertEqual(u['uniformArrayOfStructs[0].second'].location, 2)
		self.assertEqual(u['uniformArrayOfStructs[0].third'].location, 3)
		self.assertEqual(u['uniformArrayOfStructs[1].first'].location, 4)
		self.assertEqual(u['uniformArrayOfStructs[1].second'].location, 5)
		self.assertEqual(u['uniformArrayOfStructs[1].third'].location, 6)
	
	def test_base(self):
		u = self.program.uniforms['aUniform']
		self.assertNotEqual(u.location, -1)

		np_test.assert_array_equal(u.data, numpy.array([1, 2, 3]))
		u.data = numpy.array([3, 4, 5], dtype='float32')
		np_test.assert_array_equal(u.data, numpy.array([3, 4, 5]))

		with self.assertRaises(ValueError):
			u.data = numpy.array([3, 4], dtype='float32')
		with self.assertRaises(ValueError):
			u.data = numpy.array([3, 4, 5], dtype='int')
	
	def test_array_uniform(self):
		u = self.program.uniforms['matrixArrayUniform[0]']
		self.assertNotEqual(u.location, -1)

		initial = numpy.zeros((2, 4, 4), dtype='float32')
		initial[0] = numpy.eye(4)
		initial[1] = numpy.eye(4) * 2

		np_test.assert_array_equal(u.data, initial)

		new = numpy.ones((2, 4, 4), dtype='float32')
		u.data = new
		np_test.assert_array_equal(u.data, new)

		with self.assertRaises(ValueError):
			u.data = numpy.ones((3, 4, 6), dtype='float32')
		with self.assertRaises(ValueError):
			u.data = numpy.ones((2, 4, 4), dtype='float64')
		with self.assertRaises(ValueError):
			u.data = numpy.ones((4, 4), dtype='float64')

	def test_invalid_uniform(self):
		shader_files = { 'vertex': 'uniform.vert'
		               , 'fragment': 'compile.frag' }
		shaders = readShaders(**shader_files)
		uniforms = [Uniform('foobar', 'vec3')]

		program = Program(shaders, uniforms=uniforms)

		with self.assertRaises(RuntimeError):
			program.uniforms['foobar'].data
