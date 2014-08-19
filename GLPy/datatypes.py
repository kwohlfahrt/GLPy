from OpenGL import GL
from OpenGL.constants import GLboolean,GLint, GLuint, GLfloat
from itertools import product
# TODO: dobule-based types

from numpy import dtype

from util.misc import product as totalProduct

vector_sizes = range(2, 5)
data_types = ['bool', 'int', 'uint', 'float']
prefixes = { 'bool': 'b'
           , 'int': 'i'
           , 'uint': 'u'
           , 'float': '' }

base_types = {d: d for d in data_types}
base_types.update({ "{}vec{}".format(prefixes[data_type], size): base_types[data_type]
                    for data_type, size in product(data_types, vector_sizes) })
base_types.update({ "mat{}".format(size): 'float'
                    for size in vector_sizes })
base_types.update({ "mat{}x{}".format(size1, size2): 'float'
                    for size1, size2 in product(vector_sizes, repeat=2) })

uniform_types = { 'bool': 'i'
                , 'int': 'i'
                , 'uint': 'ui'
                , 'float': 'f' }

numpy_types = { 'bool': dtype('int32') # UPSTREAM: refuses to glGetUniform if GLboolean
			  , 'int': dtype('int32')
			  , 'uint': dtype('uint32')
			  , 'float': dtype('float32') }
numpy_types.update({ "{}vec{}".format(prefixes[data_type], size): dtype((numpy_types[data_type], size))
					 for data_type, size in product(data_types, vector_sizes) })
numpy_types.update({ "mat{}".format(size): dtype((numpy_types['float'], (size, size)))
					 for size in vector_sizes })
numpy_types.update({ "mat{}".format(size1, size2): dtype((numpy_types['float'], (size1, size2)))
					 for size1, size2 in product(vector_sizes, repeat=2) })

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
		return totalProduct(self.shape)

	@property
	def dtype(self):
		return numpy_types[self.type]
	
	@property
	def base_dtype(self):
		return self.dtype.subdtype[0] if self.dtype.subdtype else self.dtype

class GLSLVar(GLSLType):
	def __init__(self, name, gl_type, shape=1):
		super().__init__(gl_type=gl_type, shape=shape)
		self.name = name
	
	def __str__(self):
		array = ''.join("[{}]".format(s) for s in self.shape) if self.shape != (1,) else ""
		return "{} {}{}".format(self.type, self.name, array)

class BlockMember(GLSLVar):
	def __init__(self, block, name, gl_type, shape=1):
		super().__init__(name, gl_type, shape)
		self.block = block
	
	@classmethod
	def fromGLSLVar(cls, block, var):
		return cls(block, var.name, var.type, var.shape)

	@property
	def gl_name(self):
		return (self.name if not self.block.instance_name
		        else '.'.join((self.block.instance_name, self.name)))
	
	@property
	def api_name(self):
		return (self.name if not self.block.instance_name
		        else '.'.join((self.block.name, self.name)))
