#!/usr/bin/python3

import os, unittest

from OpenGL import GLUT
import numpy
from numpy.testing import assert_array_equal

from GLPy import ( Program, Variable, Type, VAO, VertexAttribute
                 , Buffer, ImmutableTexture )

class ContextTest(unittest.TestCase):
	def setUp(self):
		self.window_size = (400, 400)
		GLUT.glutInit()
		GLUT.glutInitContextVersion(3, 3)
		GLUT.glutInitContextProfile(GLUT.GLUT_CORE_PROFILE)
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

class ProgramTest(ContextTest):
	def setUp(self):
		super().setUp()

		shader_files = { 'vertex': 'compile.vert'
		               , 'fragment': 'compile.frag'}
		shaders = readShaders(**shader_files)
		self.program = Program(shaders)
	
	def test_compilation(self):
		pass
	
@unittest.skip('Working on buffers')
class UniformTest(unittest.TestCase):
	def setUp(self):
		shader_files = { 'vertex': 'uniform.vert'
		               , 'fragment': 'compile.frag' }
		shaders = readShaders(**shader_files)

		uniforms = [ Variable('xform', 'mat4', 1)
		           , Variable('origin', 'bool', 1)
		           , Variable('is', 'int', 3) ]
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
		new_xform = numpy.arange(16, dtype='float32')
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
		uniforms = [Variable('xform', 'mat4', 1), Variable('foobar', 'vec3', 1)]

		program = Program(shaders, uniforms=uniforms)

		with self.assertRaises(RuntimeError):
			program.uniforms['foobar'].data

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
	
	@unittest.skip('working on buffers')
	def testSampler(self):
		size = (4, 4)
		tex = ImmutableTexture(size, components=1, bits=8, normalized=True)
		data = numpy.zeros(size, dtype='uint8')
		data[1:2, 1:2] = 255;
		tex[:,:] = data
		
		shader_files = { 'vertex': 'texture.vert'
		               , 'fragment': 'texture.frag' }
		shaders = readShaders(**shader_files)

		vertex_attributes = [ Variable('position', 'vec4') ]
		vao = VAO(*vertex_attributes)

		uniform_attributes = [ Variable('tex', 'sampler2D') ]

		program = Program(shaders, attributes=vao.attributes, uniforms=uniform_attributes)

		buf = VertexBuffer(Type('vec3'))
		data = numpy.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype='int16')
		buf[...] = data
		vao[0].data = buf.blocks[0].tracks[0]

		tex.activate(1)
		program.uniforms['tex'].data = 1
