from OpenGL import GL

import numpy
from numpy import dtype
from numpy import testing as np_test

import unittest

from .test_context import ContextTest, readShaders

from GLPy import Program, Variable, Array, VAO, Buffer, VertexAttribute, Scalar
from GLPy.vertex import VAOAttribute

class VertexAttributeTest(unittest.TestCase):
	def test_indices(self):
		v = VertexAttribute('foo', 'int')
		self.assertEqual(v.indices, 1)

		v = VertexAttribute('foo', 'vec3')
		self.assertEqual(v.indices, 1)

		v = VertexAttribute('foo', 'mat4')
		self.assertEqual(v.indices, 4)

		v = VertexAttribute('foo', 'mat2x3')
		self.assertEqual(v.indices, 2)

	def test_components(self):
		v = VertexAttribute('foo', 'int')
		self.assertEqual(v.components, 1)

		v = VertexAttribute('foo', 'vec3')
		self.assertEqual(v.components, 3)

		v = VertexAttribute('foo', 'mat4')
		self.assertEqual(v.components, 4)

		v = VertexAttribute('foo', 'mat2x3')
		self.assertEqual(v.components, 3)

class VAOAttributeTest(ContextTest):
	def test_from_vertex_attrib(self):
		vao = VAO()

		va = VertexAttribute('position', 'vec4', shader_location=3)
		vs = VAOAttribute.fromVertexAttribute(vao, va)
		self.assertEqual(len(vs), 1)
		self.assertEqual(vs[0].location, 3)
		self.assertEqual(vs[0].divisor, 0)
		self.assertEqual(vs[0].components, 4)
		self.assertEqual(vs[0].scalar_type, Scalar.float)

		va = VertexAttribute('position', 'uvec2', shader_location=3)
		vs = VAOAttribute.fromVertexAttribute(vao, va)
		self.assertEqual(len(vs), 1)
		self.assertEqual(vs[0].components, 2)
		self.assertEqual(vs[0].scalar_type, Scalar.uint)

		va = VertexAttribute('position', 'int', shader_location=3)
		vs = VAOAttribute.fromVertexAttribute(vao, va, divisor=1)
		self.assertEqual(vs[0].divisor, 1)
		self.assertEqual(vs[0].components, 1)
		self.assertEqual(vs[0].scalar_type, Scalar.int)

		va = VertexAttribute('position', 'mat3x2', shader_location=3)
		vs = VAOAttribute.fromVertexAttribute(vao, va, divisor=1)
		self.assertEqual(len(vs), 3)
		for i, v in enumerate(vs):
			self.assertEqual(v.location, i + 3)
			self.assertEqual(v.components, 2)
			self.assertEqual(v.scalar_type, Scalar.float)

class VertexTest(ContextTest):
	def setUp(self):
		super().setUp()

		shader_files = { 'vertex': 'vertices.vert'
		               , 'fragment': 'vertices.frag' }
		shaders = readShaders(**shader_files)

		vertex_attribs = [ VertexAttribute('position', 'vec4')
		                 , VertexAttribute('color', 'vec3')
						 , VertexAttribute('baz', 'vec2')
		                 , VertexAttribute('foo', 'mat2x4')
		                 , VertexAttribute('bar', Array('mat3', 2)) ]
		self.program = Program(shaders, vertex_attributes=vertex_attribs)

	def tearDown(self):
		super().tearDown()
	
	def test_location(self):
		v = self.program.vertex_attributes
		for attribute in v.values():
			self.assertGreaterEqual(attribute.location, 0)

	def test_indices(self):
		v = self.program.vertex_attributes
		self.assertEqual(v['position'].indices, 1)
		self.assertEqual(v['foo'].indices, 2)
		self.assertEqual(v['bar'].indices, 6)
	
	def test_vao_attributes(self):
		v = self.program.vertex_attributes
		vao = VAO(*self.program.vertex_attributes.values())
		buf = Buffer()
		buf.target = GL.GL_ARRAY_BUFFER # FIXME
		buf[...] = numpy.zeros(10, dtype=dtype(('float32', 4)))

		t = vao[v['position'].location]
		self.assertEqual(t.location, v['position'].location)
		vao[v['position'].location].data = buf.items

		buf[...] = numpy.zeros(10, dtype=dtype(('float32', 3))) #TODO: Value error here
		t = vao[v['position'].location]
		with self.assertRaises(ValueError):
			vao[v['position'].location].data = buf.items
