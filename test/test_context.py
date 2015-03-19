#!/usr/bin/python3

import os, unittest

from OpenGL import GLUT
import numpy
from numpy.testing import assert_array_equal

from GLPy import ( Program, ImmutableTexture )

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
		self.program = Program.fromSources(shaders)
	
	def test_compilation(self):
		pass
	
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

		program = Program.fromSources(shaders, attributes=vao.attributes, uniforms=uniform_attributes)

		buf = VertexBuffer(Type('vec3'))
		data = numpy.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype='int16')
		buf[...] = data
		vao[0].data = buf.blocks[0].tracks[0]

		tex.activate(1)
		program.uniforms['tex'].data = 1
