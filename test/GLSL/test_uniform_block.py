from GLPy.GLSL.uniform_block import *
from GLPy.GLSL import Variable

import unittest

class UniformBlockTest(unittest.TestCase):
	def test_simple(self):
		block = UniformBlock('UBlock', Variable('f', 'float'), Variable('m4', 'mat4'))
	
class UniformBlockMemberTest(unittest.TestCase):
	def test_simple(self):
		block = UniformBlock('UBlock')
		member = UniformBlockMember(block, 'f', 'float', 'row_major')
