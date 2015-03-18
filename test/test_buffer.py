from OpenGL import GL
from numpy import dtype
import numpy
from numpy import testing as np_test

import unittest

from .test_context import ContextTest, readShaders

from GLPy import Buffer

pos = dtype(('float32', 3))
uv = dtype(('float32', 2))
col = dtype(('int8', 4))

class BufferItemTest(ContextTest):
	def test_buffer_item(self):
		point = dtype([('position', pos), ('UV', uv)])
		buf = Buffer()
		with buf.bind(GL.GL_ARRAY_BUFFER):
			buf[...] = dtype((point, 100))

		p = buf.items['position']
		self.assertEqual(p.offset, 0)
		self.assertEqual(p.dtype, pos)
		self.assertEqual(p.components, 3)

		u = buf.items['UV']
		self.assertEqual(u.offset, 12)
		self.assertEqual(u.dtype, uv)
		self.assertEqual(u.components, 2)

	def test_subbuffer_item(self):
		point = dtype([('position', pos), ('UV', uv)])
		buf = Buffer()
		with buf.bind(GL.GL_ARRAY_BUFFER):
			buf[...] = dtype((point, 100))
		
		p = buf[10:].items['position']
		self.assertEqual(p.offset, 200)
		self.assertEqual(p.dtype, pos)
		self.assertEqual(p.components, 3)

		u = buf[10:].items['UV']
		self.assertEqual(u.offset, 212)
		self.assertEqual(u.dtype, uv)
		self.assertEqual(u.components, 2)
	
	def test_simple_buffer(self):
		buf = Buffer()
		with buf.bind(GL.GL_ARRAY_BUFFER):
			buf[...] = dtype(('float32', (100, 3)))
		i = buf.items
		self.assertEqual(i.dtype, dtype(('float32', 3)))
		self.assertEqual(i.components, 3)

		with buf.bind(GL.GL_ARRAY_BUFFER):
			buf[...] = dtype(('float32', (100, 3)))
		i = buf.items[0]
		self.assertEqual(i.dtype, dtype(('float32')))
		self.assertEqual(i.components, 1)

class BufferIndexingTest(ContextTest):
	def test_buffer_properties(self):
		point = dtype([('position', pos), ('UV', uv)])
		buf = Buffer()
		with buf.bind(GL.GL_ARRAY_BUFFER):
			buf[...] = dtype((point, 100))
		self.assertEqual(buf.nbytes, 2000)
		self.assertEqual(buf.dtype, dtype((point, 100)))

		with buf.bind(GL.GL_ARRAY_BUFFER):
			buf[...] = dtype((point, (20, 5)))
		self.assertEqual(buf.nbytes, 2000)
		self.assertEqual(buf.dtype, dtype((point, (20, 5))))

		with self.assertRaises(IndexError):
			buf[::2]

	def test_array_buffer_array_index(self):
		point = dtype([('position', pos), ('UV', uv)])
		buf = Buffer()
		with buf.bind(GL.GL_ARRAY_BUFFER):
			buf[...] = dtype((point, 100))
		self.assertEqual(buf[0].nbytes, 20)
		self.assertEqual(buf[0].offset, 0)
		self.assertEqual(buf[0].dtype, point)
		self.assertEqual(buf[20].nbytes, 20)
		self.assertEqual(buf[20].offset, 400)
		self.assertEqual(buf[20].dtype, point)
		self.assertEqual(buf[:20].nbytes, 400)
		self.assertEqual(buf[:20].dtype, dtype((point, 20)))

		with self.assertRaises(IndexError):
			buf[100]
		with self.assertRaises(IndexError):
			buf[0, 0]
		with self.assertRaises(IndexError):
			buf[10:15, 1]
		with self.assertRaises(IndexError):
			buf[:, 1]

	def test_array_component_index(self):
		point = dtype([('position', pos), ('UV', uv)])
		buf = Buffer()
		with buf.bind(GL.GL_ARRAY_BUFFER):
			buf[...] = dtype((point, 100))
		self.assertEqual(buf[['position', 'UV']].nbytes, 2000)
		self.assertEqual(buf[['position', 'UV']].offset, 0)
		self.assertEqual(buf[['position', 'bar', 'UV', 'foo']].offset, 0)
		with self.assertRaises(IndexError):
			buf[['position']]
		with self.assertRaises(IndexError):
			buf['position']
		self.assertEqual(buf[0]['position'].dtype, dtype(('float32', (3,))))
		self.assertEqual(buf[0:1][['position']].dtype, dtype((dtype([('position', pos)]), (1,))))
		self.assertEqual(buf[0:1][['position', 'UV']].dtype, dtype((point, (1,))))
		self.assertEqual(buf[0:1]['position'].dtype, dtype(('float32', (1, 3))))
		
	def test_multidim_array_buffer_index(self):
		point = dtype([('position', pos), ('UV', uv)])
		buf = Buffer()
		with buf.bind(GL.GL_ARRAY_BUFFER):
			buf[...] = dtype((point, (20, 5)))
		
		self.assertEqual(buf[0].nbytes, 100)
		self.assertEqual(buf[0].offset, 0)
		self.assertEqual(buf[0].dtype, dtype((point, 5)))
		self.assertEqual(buf[10].nbytes, 100)
		self.assertEqual(buf[10].offset, 1000)
		self.assertEqual(buf[10].dtype, dtype((point, 5)))
		self.assertEqual(buf[0, 0].nbytes, 20)
		self.assertEqual(buf[0, 0].offset, 0)
		self.assertEqual(buf[0, 0].dtype, point)
		self.assertEqual(buf[10, 1].nbytes, 20)
		self.assertEqual(buf[10, 1].offset, 1020)
		self.assertEqual(buf[10, 1].dtype, point)

		with self.assertRaises(IndexError):
			buf[20]
		with self.assertRaises(IndexError):
			buf[0, 0, 0]

	def test_record_index(self):
		point = dtype([('position', pos), ('UV', uv)])
		buf_type = dtype([('', point, 100), ('', col, 50)])
		buf = Buffer()
		with buf.bind(GL.GL_ARRAY_BUFFER):
			buf[...] = buf_type

		self.assertIsNone(buf.stride)

		self.assertEqual(buf[0].nbytes, 2000)
		self.assertEqual(buf['f0'].nbytes, 2000)
		self.assertEqual(buf[0].offset, 0)
		self.assertEqual(buf['f0'].offset, 0)
		self.assertEqual(buf[0].stride, 20)
		self.assertEqual(buf['f0'].stride, 20)

		self.assertEqual(buf[1].nbytes, 200)
		self.assertEqual(buf['f1'].nbytes, 200)
		self.assertEqual(buf[1].offset, 2000)
		self.assertEqual(buf['f1'].offset, 2000)
		self.assertEqual(buf[1].stride, 4)
		self.assertEqual(buf['f1'].stride, 4)

		with self.assertRaises(IndexError):
			buf['f0', 'f1']
		with self.assertRaises(IndexError):
			buf[['f0', 'f1']]

class BufferSettingTest(ContextTest):
	def test_buffer_set(self):
		point = dtype([('position', pos), ('UV', uv)])
		buf = Buffer()
		with buf.bind(GL.GL_ARRAY_BUFFER):
			buf[...] = numpy.zeros((20, 5), dtype=point)

	def test_buffer_change(self):
		point = dtype([('position', pos), ('UV', uv)])
		buf = Buffer()
		with buf.bind(GL.GL_ARRAY_BUFFER):
			buf[...] = numpy.zeros((20, 5), dtype=point)
			buf.data = numpy.ones((20, 5), dtype=point)
			with self.assertRaises(ValueError):
				buf.data = numpy.ones((20, 4), dtype=point)

	def test_buffer_get(self):
		point = dtype([('position', pos), ('UV', uv)])
		buf = Buffer()
		with buf.bind(GL.GL_ARRAY_BUFFER):
			buf[...] = numpy.zeros((20, 5), dtype=point)
			np_test.assert_equal(buf.data, numpy.zeros((20, 5), dtype=point))

	def test_sub_buffer_set(self):
		point = dtype([('position', pos), ('UV', uv)])
		buf = Buffer()
		with buf.bind(GL.GL_ARRAY_BUFFER):
			buf[...] = numpy.zeros((20, 5), dtype=point)
			buf[0].data = numpy.ones(5, dtype=point)
			buf[0][:-1].data = numpy.zeros(4, dtype=point)

	def test_sub_buffer_get(self):
		point = dtype([('position', pos), ('UV', uv)])
		buf = Buffer()
		with buf.bind(GL.GL_ARRAY_BUFFER):
			buf[...] = numpy.zeros((20, 5), dtype=point)
			np_test.assert_equal(buf[0].data, numpy.zeros(5, dtype=point))

			buf[...] = numpy.ones((20, 5), dtype=point)
			np_test.assert_equal(buf[0].data, numpy.ones(5, dtype=point))

class BufferMapTest(ContextTest):
	def test_buffer_map(self):
		point = dtype([('position', pos), ('UV', uv)])
		buf = Buffer()
		with buf.bind(GL.GL_ARRAY_BUFFER):
			buf[...] = numpy.zeros((20, 5), dtype=point)

			m = buf.map()
		self.assertTrue(buf.mapped)
		np_test.assert_equal(m, numpy.zeros((20, 5), dtype=point))
		m[0] = 1
		with buf.bind(GL.GL_ARRAY_BUFFER):
			buf.unmap()
		self.assertFalse(buf.mapped)

		with buf.bind(GL.GL_ARRAY_BUFFER):
			m = buf.map(GL.GL_MAP_READ_BIT)
		np_test.assert_equal(m[0], numpy.ones(5, dtype=point))
		np_test.assert_equal(m[1:], numpy.zeros((19, 5), dtype=point))
		with buf.bind(GL.GL_ARRAY_BUFFER):
			buf.unmap()

	def test_subbuffer_map(self):
		point = dtype([('position', pos), ('UV', uv)])
		buf = Buffer()
		with buf.bind(GL.GL_ARRAY_BUFFER):
			buf[...] = numpy.zeros((20, 5), dtype=point)

			m = buf[0].map()
		self.assertTrue(buf.mapped)
		np_test.assert_equal(m, numpy.zeros(5, dtype=point))
		m[0] = numpy.ones(1, dtype=point)
		with buf.bind(GL.GL_ARRAY_BUFFER):
			buf.unmap()
		self.assertFalse(buf.mapped)
		
		with buf.bind(GL.GL_ARRAY_BUFFER):
			m = buf[0].map()
		np_test.assert_equal(m[0], numpy.ones(1, dtype=point))
		np_test.assert_equal(m[1:], numpy.zeros(4, dtype=point))
		with buf.bind(GL.GL_ARRAY_BUFFER):
			buf.unmap()

	@unittest.skip("TODO")
	def test_buffer_unmap_invalidates(self):
		point = dtype([('position', pos), ('UV', uv)])
		buf = Buffer()
		buf.target = GL.GL_ARRAY_BUFFER
		buf[...] = numpy.zeros((20, 5), dtype=point)

		m = buf.map()
		buf.unmap()

		with self.assertRaises(IndexError):
			m[0]
		with self.assertRaises(IndexError):
			m[0] = 1

	def test_flush(self):
		point = dtype([('position', pos), ('UV', uv)])
		buf = Buffer()
		with buf.bind(GL.GL_ARRAY_BUFFER):
			buf[...] = numpy.zeros((20, 5), dtype=point)

			m = buf.map(GL.GL_MAP_WRITE_BIT | GL.GL_MAP_FLUSH_EXPLICIT_BIT)
			m.flush()
