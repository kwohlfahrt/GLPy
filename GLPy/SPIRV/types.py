from itertools import repeat, product as cartesian
from numpy import dtype
from typing import Tuple

from .parse.enums import *

class SPIRVType(type):
    pass

class PrimitiveType(SPIRVType):
    '''A scalar type, or vector or matrix thereof'''
    def __str__(self):
        return self.__name__.lower()

class ScalarType(PrimitiveType):
    '''A floating-point, integer or boolean type'''
    pass

class TypeFloat(ScalarType):
    def __new__(cls, bits: int):
        name = 'Float{}'.format(bits)
        return super().__new__(cls, name, (), {})
    def __init__(self, bits: int):
        self.bits = bits

    @property
    def machine_type(self):
        return dtype("float{}".format(self.bits))
    @property
    def prefix(self) -> str:
        if bits == 16:
            return 'h'
        elif bits == 32:
            return 'f'
        elif bits == 64:
            return 'd'
        else:
            return 'f{}'.format(self.bits)

class TypeInt(ScalarType):
    def __new__(cls, bits: int, signed: bool):
        name = '{}int{}'.format('' if signed else 'u', bits).capitalize()
        return super().__new__(cls, name, (), {})
    def __init__(self, bits: int, signed: bool):
        self.bits = bits
        self.signed = signed

    @property
    def machine_type(self):
        return dtype("{}int{}".format('' if self.signed else 'u', self.bits))
    @property
    def prefix(self) -> str:
        letter = 'i' if self.signed else 'u'
        if self.bits == 32:
            return letter
        else:
            return '{}{}'.format(letter, self.bits)

class TypeBool(ScalarType):
    prefix = 'b'
    # No bit pattern or physical size defined, not allowed in visible memory

    def __new__(cls):
        return super().__new__(cls, 'Bool', (), {})

class TypeVector(PrimitiveType):
    def __new__(cls, component_type: ScalarType, component_count: int):
        if component_count < 2:
            raise ValueError("Vector types must have at least 2 components")
        name = '{}vec{}'.format(component_type.prefix, component_count).capitalize()
        return super().__new__(cls, name, (), {})
    def __init__(self, component_type: ScalarType, component_count: int):
        self.component_type = component_type
        self.component_count = component_count

    def __hash__(self):
        return hash((self.component_type, self.component_count))
    def __eq__(self, other):
        return (self.component_type == other.component_type
                and self.component_count == other.component_count)

class TypeMatrix(PrimitiveType):
    def __new__(cls, column_type: TypeVector, column_count: int):
        if column_count < 2:
            raise ValueError("Vector types must have at least 2 components")
        name = '{}mat{}x{}'.format(column_type.component_type.prefix, column_count,
                                   column_type.component_count).capitalize()
        return super().__new__(cls, name, (), {})
    def __init__(self, column_type: TypeVector, column_count: int):
        self.column_type = column_type
        self.column_count = column_count

    def __hash__(self):
        return hash((self.columns_type, self.columns_count))
    def __eq__(self, other):
        return (self.columns_type == other.columns_type
                and self.columns_count == other.columns_count)
    def __len__(self):
        return self.column_count
    def __getitem__(self, idx):
        if not (0 <= idx < len(self)):
            raise IndexError("Index {} out of range for matrix with {} columns"
                             .format(idx, len(self)))
        return self.column_type

class TypeArray(SPIRVType):
    def __new__(cls, element_type: SPIRVType, length: int):
        if length < 0:
            raise ValueError("Array types must have length of 1 or greater")
        name = '_Array_'.join((element_type.__qualname__, str(length)))
        return super().__new__(cls, name, (), {})
    def __init__(self, element_type: SPIRVType, length: int):
        self.element_type = element_type
        self.length = length

    def __hash__(self):
        return hash((self.element_type, self.length))
    def __eq__(self, other):
        return (self.element_type == other.element_type
                and self.length == other.length)
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        if not (0 <= idx < len(self)):
            raise IndexError("Index {} out of range for array with {} elements"
                             .format(idx, len(self)))
        return self.element_type
    def __str__(self):
        return '{}[{}]'.format(self.base, ']['.join(map(str, self.full_shape)))

    @property
    def full_shape(self):
        return (len(self),) + getattr(self.element_type, 'full_shape', ())
    @property
    def base(self):
        return getattr(self.element_type, 'base', self.element_type)

class TypeStruct(SPIRVType):
    def __new__(cls, *contents: Tuple[SPIRVType, ...]):
        # FIXME: Find something that would be a legal identifier name?
        name = 'Struct({})'.format(' '.join(map(repr, contents)))
        return super().__new__(cls, name, (), {})
    def __init__(self, *contents: Tuple[SPIRVType, ...]):
        self.contents = contents

    def __hash__(self):
        return hash(self.contents)
    def __eq__(self, other):
        return self.contents == other.contents

class TypePointer(SPIRVType):
    def __new__(cls, target: SPIRVType, storage: StorageClass):
        name = "Pointer({}, storage={})".format(repr(target), storage)
        return super().__new__(cls, name, (), {})
    def __init__(self, target: SPIRVType, storage: StorageClass):
        self.target = target
        self.storage = storage

    def __hash__(self):
        return hash((self.target, self.storage))
    def __eq__(self):
        return (self.target == other.target
                and self.storage == other.storage)
    def __str__(self):
        return "ptr[{}]".format(str(self.target))

def variable(result_type: TypePointer, storage: StorageClass):
    if result_type.storage != storage:
        raise ValueError("Cannot point to variable with storage {} using pointer with storage {}"
                         .format(result_type.storage, storage))
    return result_type()

if __name__ == '__main__':
    Float = TypeFloat(32)
    print(Float, repr(Float), Float.machine_type)
    Double = TypeFloat(64)
    print(Double, repr(Double), Double.machine_type)
    Int = TypeInt(32, True)
    print(Int, repr(Int), Int.machine_type)
    Uint = TypeInt(32, False)
    print(Uint, repr(Uint), Uint.machine_type)
    Bool = TypeBool()
    print(Bool, repr(Bool), Bool)

    Fvec3 = TypeVector(Float, 3)
    print(Fvec3, repr(Fvec3))
    Dvec2 = TypeVector(Double, 2)
    print(Dvec2, repr(Dvec2))

    Fmat4x3 = TypeMatrix(Fvec3, 4)
    print(Fmat4x3, repr(Fmat4x3))

    VecArray = TypeArray(Fvec3, 3)
    print(VecArray, repr(VecArray))
    ArrayArray = TypeArray(VecArray, 6)
    print(ArrayArray, repr(ArrayArray))

    FloatPtr = TypePointer(Float, StorageClass.UniformConstant)
    print(FloatPtr)
    print(variable(FloatPtr, StorageClass.UniformConstant))