from OpenGL import GL

import numpy
from numpy import dtype
from numpy import testing as np_test

import unittest

from .test_context import ContextTest, readShaders

from GLPy.GLSL import Variable, Array, Scalar, VertexAttribute
from GLPy import Program, VAO, Buffer
from GLPy.vertex import VAOAttribute

class VAOAttributeTest(ContextTest):
	def test_from_vertex_attrib(self):
		vao = VAO()

		va = VertexAttribute('position', 'vec4', location=3)
		vs = VAOAttribute.fromVertexAttribute(vao, va)
		self.assertEqual(len(vs), 1)
		self.assertEqual(vs[0].location, 3)
		self.assertEqual(vs[0].divisor, 0)
		self.assertEqual(vs[0].components, 4)
		self.assertEqual(vs[0].scalar_type, Scalar.float)

		va = VertexAttribute('position', 'uvec2', location=3)
		vs = VAOAttribute.fromVertexAttribute(vao, va)
		self.assertEqual(len(vs), 1)
		self.assertEqual(vs[0].components, 2)
		self.assertEqual(vs[0].scalar_type, Scalar.uint)

		va = VertexAttribute('position', 'int', location=3)
		vs = VAOAttribute.fromVertexAttribute(vao, va, divisor=1)
		self.assertEqual(vs[0].divisor, 1)
		self.assertEqual(vs[0].components, 1)
		self.assertEqual(vs[0].scalar_type, Scalar.int)

		va = VertexAttribute('position', 'mat3x2', location=3)
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
		with buf.bind(GL.GL_ARRAY_BUFFER):
			buf[...] = numpy.zeros(10, dtype=dtype(('float32', 4)))

		t = vao[v['position'].location]
		self.assertEqual(t.location, v['position'].location)
		vao[v['position'].location].data = buf.items

	@unittest.skip("TODO")
	def test_vao_dtype_fail(self):
		v = self.program.vertex_attributes
		vao = VAO(*self.program.vertex_attributes.values())
		buf = Buffer()
		with buf.bind(GL.GL_ARRAY_BUFFER):
			buf[...] = numpy.zeros(10, dtype=dtype(('float32', 3)))

		with self.assertRaises(ValueError):
			vao[v['position'].location].data = buf.items

		with buf.bind(GL.GL_ARRAY_BUFFER):
			buf[...] = numpy.zeros(10, dtype=dtype(('float32', 4)))
		vao[v['position'].location].data = buf.items

		with buf.bind(GL.GL_ARRAY_BUFFER):
			with self.assertRaises(ValueError):
				buf[...] = numpy.zeros(10, dtype=dtype(('float32', 3)))
