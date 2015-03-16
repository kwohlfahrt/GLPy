'''A module containing types to represent GLSL code. All classes in this module
should function using only information found in shaders.'''

from .datatypes import ( Scalar, Vector, Matrix, Sampler, BasicType
                       , Array, Struct )
from .variable import Variable
from .interface_block import BlockLayout, MatrixLayout
from .uniform_block import UniformBlock, UniformBlockMember
