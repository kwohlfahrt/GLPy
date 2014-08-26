#!/usr/bin/python3

from math import radians
import os, unittest

from OpenGL import GLUT, GL
import numpy
from numpy.testing import assert_array_equal

from GLPy import Program, GLSLVar, GLSLType, VAO, VertexBuffer, UniformBlock, UniformBuffer, ImmutableTexture

class ContextTest(unittest.TestCase):
	def setUp(self):
		self.window_size = (400, 400)
		GLUT.glutInit()
		GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA)
		GLUT.glutInitWindowSize(*self.window_size)
		self.window = GLUT.glutCreateWindow("GLPy Test")
	
	def tearDown(self):
		GLUT.glutDestroyWindow(self.window)

	def test_context(self):
		# See if setUp and tearDown succeed
		pass

def readShaders(**shader_paths):
	shaders = {}
	for shader, path in shader_paths.items():
		path = os.path.join(os.path.dirname(__file__), path)
		with open(path) as f:
			shaders[shader] = f.read()
	return shaders

class ProgramTest(unittest.TestCase):
	def setUp(self):
		shader_files = { 'vertex': 'compile.vert'
		               , 'fragment': 'compile.frag'}
		shaders = readShaders(**shader_files)
		ContextTest.setUp(self)
		self.program = Program(shaders)
	
	def tearDown(self):
		ContextTest.tearDown(self)

	def test_compilation(self):
		pass
	
class UniformTest(unittest.TestCase):
	def setUp(self):
		shader_files = { 'vertex': 'uniform.vert'
		               , 'fragment': 'compile.frag' }
		shaders = readShaders(**shader_files)

		uniforms = [ GLSLVar('xform', 'mat4', 1)
		           , GLSLVar('origin', 'bool', 1)
				   , GLSLVar('is', 'int', 3) ]
		ContextTest.setUp(self)
		self.program = Program(shaders, uniforms=uniforms)

	def tearDown(self):
		ContextTest.tearDown(self)
	
	def test_boolean(self):
		self.assertFalse(self.program.uniforms['origin'].data)
		self.program.uniforms['origin'].data = True
		self.assertTrue(self.program.uniforms['origin'].data)
	
	def test_matrix(self):
		assert_array_equal( self.program.uniforms['xform'].data
		                  , numpy.identity(4))
		new_xform = numpy.arange(16)
		new_xform.shape = (4, 4)
		self.program.uniforms['xform'].data = new_xform
		assert_array_equal( self.program.uniforms['xform'].data
		                  , new_xform)
	
	def test_array(self):
		old_is = numpy.array([1, 2, 3], dtype='int32')
		assert_array_equal(self.program.uniforms['is'].data, old_is)
		new_is = numpy.array([6, 2, 0], dtype='int32')
		self.program.uniforms['is'] = new_is
		assert_array_equal(self.program.uniforms['is'].data, new_is)
	
	def test_invalid_uniform(self):
		shader_files = { 'vertex': 'uniform.vert'
		               , 'fragment': 'compile.frag' }
		shaders = readShaders(**shader_files)
		uniforms = [GLSLVar('xform', 'mat4', 1), GLSLVar('foobar', 'vec3', 1)]

		program = Program(shaders, uniforms=uniforms)

		with self.assertRaises(RuntimeError):
			program.uniforms['foobar'].data

# TODO: Test matrix vertex attributes
class VertexBufferTest(unittest.TestCase):
	def setUp(self):
		self.buffer_contents = [ GLSLType('vec3')
		                       , GLSLType('vec3') ]
		ContextTest.setUp(self)

	def tearDown(self):
		ContextTest.tearDown(self)
	
	def test_compile(self):
		pass
	
	def test_simple_buffer(self):
		buf = VertexBuffer(GLSLType('vec3'))
		data = numpy.array([[1, 1, 1], [1, 2, 3], [1, 2, 1]], dtype='float32')
		buf[...] = data,
		self.assertEqual(len(buf), 3)
		buf.blocks[0][...] = data
		self.assertEqual(len(buf), 3)
		data = numpy.array([[1, 1, 1, 2], [1, 2, 3, 2], [1, 2, 1, 2]], dtype='float32')
		with self.assertRaises(ValueError):
			buf[...] = data,
		with self.assertRaises(ValueError):
			buf.blocks[0][:] = data
	
	def test_struct_buffer(self):
		buf = VertexBuffer([GLSLType('vec3'), GLSLType('float')])
		dt = numpy.dtype([('', 'float32', 3), ('', 'float32', 1)])
		data = numpy.array([([1, 1, 1], 3)
		                   ,([1, 2, 3], 0)
						   ,([4, 0, 0], 2)]
						  , dtype=dt)
		buf[...] = data,
		self.assertEqual(len(buf), 3)
		buf.blocks[0][:] = data
		self.assertEqual(len(buf), 3)
		dt = numpy.dtype([('', 'float32', 4), ('', 'float32', 1)])
		data = numpy.array([([1, 1, 1, 0], 3)
		                   ,([1, 2, 3, 0], 0)
						   ,([4, 0, 0, 0], 2)]
						  , dtype=dt)
		with self.assertRaises(ValueError):
			buf[...] = data,
		with self.assertRaises(ValueError):
			buf.blocks[0][...] = data

	def test_array_struct_buffer(self):
		buf = VertexBuffer([GLSLType('vec3'), GLSLType('float', shape=(2,))])
		dt = numpy.dtype([('', 'float32', 3), ('', 'float32', 2)])
		data = numpy.array([([1, 1, 1], [3, 4])
		                   ,([1, 2, 3], [0, 3])
						   ,([4, 0, 0], [2, 2])]
						  , dtype=dt)
		buf[...] = data,
		self.assertEqual(len(buf), 3)
		buf.blocks[0][...] = data
		self.assertEqual(len(buf), 3)

class VertexAttributeTest(unittest.TestCase):
	def setUp(self):
		shader_files = { 'vertex': 'vertices.vert'
		               , 'fragment': 'vertices.frag' }
		shaders = readShaders(**shader_files)

		vertex_attributes = [ GLSLVar('position', 'vec4')
							, GLSLVar('color', 'vec3') ]

		ContextTest.setUp(self)
		self.vao = VAO(*vertex_attributes)
		self.program = Program(shaders, attributes=self.vao.attributes)

	def tearDown(self):
		ContextTest.tearDown(self)
	
	def test_position(self):
		buf = VertexBuffer(GLSLType('vec3'))
		data = numpy.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype='int16')
		buf[...] = data
		self.vao[0].data = buf.blocks[0].tracks[0]

class UniformBlockTest(unittest.TestCase):
	def setUp(self):
		shader_files = { 'vertex': 'uniform_block.vert'
		               , 'fragment': 'compile.frag' }
		shaders = readShaders(**shader_files)

		uniforms = [ GLSLVar('xform', 'mat4', 1)
		           , GLSLVar('origin', 'bool', 1) ]
		ContextTest.setUp(self)
		self.program = Program(shaders)
		self.uniform_block = UniformBlock(1, self.program
		                                 , "Projection"
										 , *uniforms)

	def tearDown(self):
		ContextTest.tearDown(self)
	
	def test_get_block_index(self):
		self.assertEqual(self.uniform_block.index, 0)
	
	def test_get_member_indices(self):
		indices = [m.index for m in self.uniform_block.members]
		self.assertIn(0, indices)
		self.assertIn(1, indices)
	
	def test_get_member_offsets(self):
		offsets = [m.offset for m in self.uniform_block.members]
		self.assertIn(0, offsets)
		self.assertIn(64, offsets)
	
	def test_buffer(self):
		buf = UniformBuffer(self.uniform_block)
		buf.blocks[0][1] = numpy.asarray(True, dtype='float32')
		buf.blocks[0][0] = numpy.eye(4, dtype='float32')
	
	def test_buffer_fail(self):
		buf = UniformBuffer(self.uniform_block)
		with self.assertRaises(ValueError):
			buf.blocks[0][0] = numpy.asarray(True, dtype='float32')
		with self.assertRaises(ValueError):
			buf.blocks[0][0] = numpy.eye(3, dtype='float32')
	
	def test_buffer_record(self):
		buf = UniformBuffer(self.uniform_block)
		a = numpy.arange(16, dtype='float32')
		dt = numpy.dtype([('', 'float32', (4, 4))])
		a.dtype = dt
		buf.blocks[0][0] = a

	def test_buffer_record_fail(self):
		buf = UniformBuffer(self.uniform_block)
		a = numpy.arange(12, dtype='float32')
		dt = numpy.dtype([('', 'float32', (3, 4))])
		a.dtype = dt
		with self.assertRaises(ValueError):
			buf.blocks[0][0] = a

class TextureTest(unittest.TestCase):
	def setUp(self):
		ContextTest.setUp(self)
	
	def tearDown(self):
		ContextTest.tearDown(self)
	
	def testTexture(self):
		size = (4, 4)
		tex = ImmutableTexture(size, components=1, bits=8, normalized=False)
		data = numpy.zeros(size, dtype='uint8')
		data[1:2, 1:2] = 1;
		tex[:,:] = data
	
	def testSampler(self):
		size = (4, 4)
		tex = ImmutableTexture(size, components=1, bits=8, normalized=True)
		data = numpy.zeros(size, dtype='uint8')
		data[1:2, 1:2] = 255;
		tex[:,:] = data
		
		shader_files = { 'vertex': 'texture.vert'
		               , 'fragment': 'texture.frag' }
		shaders = readShaders(**shader_files)

		vertex_attributes = [ GLSLVar('position', 'vec4') ]
		vao = VAO(*vertex_attributes)

		uniform_attributes = [ GLSLVar('tex', 'sampler2D') ]

		program = Program(shaders, attributes=vao.attributes, uniforms=uniform_attributes)

		buf = VertexBuffer(GLSLType('vec3'))
		data = numpy.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype='int16')
		buf[...] = data
		vao[0].data = buf.blocks[0].tracks[0]

		tex.activate(1)
		program.uniforms['tex'].data = 1
