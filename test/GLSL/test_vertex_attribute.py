from GLPy.GLSL.vertex_attribute import *

import unittest

class VertexAttributeTest(unittest.TestCase):
	def test_indexing(self):
		v = VertexAttribute('foo', 'mat4')
		self.assertEqual(v[2], VertexAttribute('foo[2]', 'vec4'))

		v = VertexAttribute('foo', Array('mat4', 3), location=3)
		self.assertEqual(v[2], VertexAttribute('foo[2]', 'mat4', location=11))
