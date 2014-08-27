from OpenGL import GL
from OpenGL.constants import GLboolean,GLint, GLuint, GLfloat
from itertools import product as cartesian
# TODO: dobule-based types

from numpy import dtype

from util.misc import product

vector_sizes = range(2, 5)
data_types = ['bool', 'int', 'uint', 'float']
sampler_types = { '1D': GL.GL_TEXTURE_1D
                , '2D': GL.GL_TEXTURE_2D
				, '3D': GL.GL_TEXTURE_3D }

prefixes = { 'bool': 'b'
           , 'int': 'i'
           , 'uint': 'u'
           , 'float': '' }

base_types = {d: d for d in data_types}
base_types.update({ "{}vec{}".format(prefixes[data_type], size): base_types[data_type]
                    for data_type, size in cartesian(data_types, vector_sizes) })
base_types.update({ "mat{}".format(size): 'float'
                    for size in vector_sizes })
base_types.update({ "mat{}x{}".format(size1, size2): 'float'
                    for size1, size2 in cartesian(vector_sizes, repeat=2) })
base_types.update({ "{}sampler{}".format(data_type, sampler_type): 'int'
                    for data_type, sampler_type in cartesian(['', 'i', 'u'], sampler_types)})

numpy_types = { 'bool': dtype('int32') # UPSTREAM: refuses to glGetUniform if GLboolean
			  , 'int': dtype('int32')
			  , 'uint': dtype('uint32')
			  , 'float': dtype('float32') }
numpy_types.update({ "{}vec{}".format(prefixes[data_type], size): dtype((numpy_types[data_type], size))
					 for data_type, size in cartesian(data_types, vector_sizes) })
numpy_types.update({ "mat{}".format(size): dtype((numpy_types['float'], (size, size)))
					 for size in vector_sizes })
numpy_types.update({ "mat{}".format(size1, size2): dtype((numpy_types['float'], (size1, size2)))
					 for size1, size2 in cartesian(vector_sizes, repeat=2) })

buf_types = { GL.GL_BYTE: dtype('int8')
            , GL.GL_UNSIGNED_BYTE: dtype('uint8')
            , GL.GL_SHORT: dtype('int16')
            , GL.GL_UNSIGNED_SHORT: dtype('uint16')
            , GL.GL_INT: dtype('int32')
            , GL.GL_UNSIGNED_INT: dtype('uint32')
            , GL.GL_HALF_FLOAT: dtype('float16')
            , GL.GL_FLOAT: dtype('float32')
            , GL.GL_DOUBLE: dtype('float64') }

gl_types = {v: k for k, v in buf_types.items()}
gl_integer_types = { GL.GL_BYTE, GL.GL_UNSIGNED_BYTE, GL.GL_SHORT, GL.GL_UNSIGNED_SHORT, GL.GL_INT, GL.GL_UNSIGNED_INT }
gl_float_types = { GL.GL_HALF_FLOAT, GL.GL_FLOAT }
gl_double_types = { GL.GL_DOUBLE }

class GLSLType:
	'''A class to represent an OpenGL data type, not necessarily a named variable.

	:param str gl_type: The OpenGL data type (e.g. ``vec3``)
	:param shape: The shape of the data type if it is an array
	:type shape: :py:obj:`int` or [:py:obj:`int`]
	'''
	def __init__(self, gl_type, shape=1):
		# TODO: Use enums when Python-3.4 is stable
		if gl_type not in base_types:
			raise ValueError("Invalid OpenGL type: {}".format(repr(gl_type)))
		self.type = gl_type
		try:
			self.shape = tuple(shape)
		except TypeError:
			self.shape = (shape,)
	
	def __str__(self):
		array = ''.join("[{}]".format(s) for s in self.shape) if self.shape != (1,) else ""
		return "{}{}".format(self.type, array)
	
	def __repr__(self):
		return "{}({})".format(type(self).__name__, self)
	
	@property
	def count(self):
		'''The total number of elements in the variable'''
		return product(self.shape)

	@property
	def dtype(self):
		'''The numpy dtype corresponding to one element of this variable'''
		return numpy_types[self.type]

class GLSLVar(GLSLType):
	'''A class to represent a named OpenGL variable.

	:param str name: The name of the OpenGL variable
	:param str gl_type: The OpenGL data type (e.g. ``vec3``)
	:param shape: The shape of the data type if it is an array
	:type shape: :py:obj:`int` or [:py:obj:`int`]
	'''

	def __init__(self, name, gl_type, shape=1):
		super().__init__(gl_type=gl_type, shape=shape)
		self.name = name

	@classmethod
	def fromGLSLType(cls, name, glsl_type):
		'''Construct from a name and an instance of :py:class:`.GLSLType`

		:param str name: the name of the variable
		:param glsl_type: The type of the variable
		:type glsl_type: :py:class:`.GLSLType`
		'''
		return cls(name, glsl_type.type, glsl_type.shape)

	def __str__(self):
		array = ''.join("[{}]".format(s) for s in self.shape) if self.shape != (1,) else ""
		return "{} {}{}".format(self.type, self.name, array)

class BlockMember(GLSLVar):
	'''A variable that is a member of a data block (e.g. a uniform block)'''
	def __init__(self, block, name, gl_type, shape=1):
		super().__init__(name, gl_type, shape)
		self.block = block

	@classmethod
	def fromGLSLVar(cls, block, var):
		'''Construct from a block :py:class:`.GLSLVar`

		:param str block: the block the variable belongs to
		:param var: the variable describing the block member
		:type var: :py:class:`.GLSLVar`
		'''
		return cls(block, var.name, var.type, var.shape)

	@property
	def gl_name(self):
		'''The string used to refer to the block member in a shader'''
		return (self.name if not self.block.instance_name
		        else '.'.join((self.block.instance_name, self.name)))

	@property
	def api_name(self):
		'''The string used to refer to the block member in the OpenGL AP'''
		return (self.name if not self.block.instance_name
		        else '.'.join((self.block.name, self.name)))
