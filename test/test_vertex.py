from OpenGL import GL

import numpy
from numpy import dtype
from numpy import testing as np_test

import unittest

from .test_context import ContextTest, readShaders

from GLPy.GLSL import Variable, Array, Scalar, VertexAttribute
from GLPy import Program, VAO, Buffer
from GLPy.vertex import VAOAttribute

class VAOTest(ContextTest):
	def test_overlapping(self):
		with self.assertRaises(ValueError):
			VAO(VertexAttribute('m4', 'mat4', location=0),
			    VertexAttribute('v3', 'vec3', location=2))

	def test_indexing(self):
		vao = VAO(VertexAttribute('m4', 'mat4', location=3))
		self.assertEqual(vao['m4'], VAOAttribute(vao, 3, 'mat4'))
		self.assertEqual(vao['m4'][1], VAOAttribute(vao, 4, 'vec4'))

		vao = VAO(VertexAttribute('2m4', Array('mat4', 2), location=3))
		self.assertEqual(vao['2m4'], VAOAttribute(vao, 3, Array('mat4', 2)))
		self.assertEqual(vao['2m4'][1], VAOAttribute(vao, 7, 'mat4'))

	def test_divisor(self):
		vao = VAO(VertexAttribute('m4', 'mat4', location=3))
		self.assertEqual(vao['m4'].divisor, 0)
		vao['m4'].divisor = 1
		self.assertEqual(vao['m4'].divisor, 1)
		self.assertEqual(vao['m4'][2].divisor, 1)

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
		self.vao = VAO(*self.program.vertex_attributes.values())

	def tearDown(self):
		super().tearDown()
	
	def test_location(self):
		v = self.program.vertex_attributes
		for attribute in v.values():
			self.assertGreaterEqual(attribute.location, 0)
	
	def test_vao_attributes(self):
		buf = Buffer()
		with buf.bind(GL.GL_ARRAY_BUFFER):
			buf[...] = numpy.zeros(10, dtype=dtype(('float32', 4)))

		self.assertEqual(self.vao['position'].location,
		                 self.program.vertex_attributes['position'].location)
		self.vao['position'].data = buf.items

	def test_vao_dtype_fail(self):
		buf = Buffer()
		with buf.bind(GL.GL_ARRAY_BUFFER):
			buf[...] = numpy.zeros(10, dtype=dtype(('float32', 3)))

		with self.assertRaises(ValueError):
			self.vao['position'].data = buf.items

		with buf.bind(GL.GL_ARRAY_BUFFER):
			buf[...] = numpy.zeros(10, dtype=dtype(('float32', 4)))
		self.vao['position'].data = buf.items

	@unittest.skip("TODO")
	def test_buffer_reassign_fail(self):
		buf = Buffer()
		with buf.bind(GL.GL_ARRAY_BUFFER):
			buf[...] = numpy.zeros(10, dtype=dtype(('float32', 4)))
		self.vao['position'].data = buf.items
		with buf.bind(GL.GL_ARRAY_BUFFER):
			with self.assertRaises(ValueError):
				buf[...] = numpy.zeros(10, dtype=dtype(('float32', 3)))
