from GLPy.GLSL.vertex_attribute import *

import unittest

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
