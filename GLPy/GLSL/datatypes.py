from itertools import repeat, product as cartesian
from collections import OrderedDict

from numpy import dtype

from enum import Enum

class _DatatypeMeta(type):
    def __str__(self):
        return self.__name__.lower()
    def __mul__(self, num):
        if num < 1:
            raise ValueError("Cannot have arrays with length less than 1")
        if num % 1:
            raise ValueError("Cannot have arrays of non-integer length")
        return _ArrayMeta(self, num)

class _Datatype:
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return ' '.join((str(type(self)), self.name))
    def __repr__(self):
        return '{}(name="{}")'.format(type(self).__name__, self.name)

    @property
    def machine_type(self):
        return dtype((self.scalar_type.machine_type, self.length))

class _ScalarMeta(_DatatypeMeta):
    def __new__(cls, name, machine_type, prefix):
        return super().__new__(cls, name, (_Datatype,), {})
    def __init__(self, name, machine_type, prefix):
        self.prefix = prefix
        self.machine_type = machine_type

    def __hash__(self):
        return hash(self.machine_type)
    def __eq__(self, other):
        return self.machine_type == other.machine_type

class _VectorMeta(_DatatypeMeta):
    def __new__(cls, scalar_type, length):
        name = '{}vec{}'.format(scalar_type.prefix, length).capitalize()
        return super().__new__(cls, name, (_Datatype,), {})
    def __init__(self, scalar_type, length):
        self.scalar_type = scalar_type
        self.length = length

    def __hash__(self):
        return hash((self.scalar_type, self.length))
    def __eq__(self, other):
        return (self.scalar_type == other.scalar_type
                and self.length == other.length)

    @property
    def machine_type(self):
        return dtype((self.scalar_type.machine_type, self.length))

class _IndexableType(_Datatype):
    def __len__(self):
        return len(type(self))
    def __getitem__(self, idx):
        name = "{}[{}]".format(self.name, idx)
        return type(self)[idx](name)

class _Matrix(_IndexableType):
    @property
    def shape(self):
        return type(self).shape

class _MatrixMeta(_DatatypeMeta):
    def __new__(cls, scalar_type, columns, rows):
        name = '{}mat{}x{}'.format(scalar_type.prefix, columns, rows).capitalize()
        return super().__new__(cls, name, (_Matrix,), {})
    def __init__(self, scalar_type, columns, rows):
        self.scalar_type = scalar_type
        self.columns = columns
        self.rows = rows

    def __len__(self):
        return self.columns
    def __getitem__(self, idx):
        if not (0 <= idx < len(self)):
            raise IndexError("Index {} out of range for a matrix with {} columns"
                             .format(idx, self.columns))
        return _VectorMeta(self.scalar_type, self.rows)
    def __hash__(self):
        return hash((self.scalar_type, self.columns, self.rows))
    def __eq__(self, other):
        return (self.scalar_type == other.scalar_type
                and self.columns == other.columns
                and self.rows == other.rows)

    @property
    def shape(self):
        return (self.columns, self.rows)
    @property
    def machine_type(self):
        # FIXME: Return ((scalar, cols), rows) instead?
        # Consistency with arrays, perhaps uniform memory?
        return dtype((self.scalar_type.machine_type, self.shape))

def formatShape(shape):
    return ''.join('[{}]'.format(s) for s in shape)

class _ArrayMeta(_DatatypeMeta):
    def __new__(cls, element, length):
        name = "{}_Array_{}".format(element.__qualname__, length)
        return super().__new__(cls, name, (_IndexableType,), {})
    def __init__(self, element, length):
        self.element = element
        self.length = length

    def __str__(self):
        return ''.join((str(self.base), formatShape(self.full_shape)))
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        if not (0 <= idx < len(self)):
            raise IndexError("Index {} out of range for array with length {}"
                                .format(idx, len(self)))
        return self.element
    def __hash__(self):
        return hash((self.element, self.length))
    def __eq__(self, other):
        return (self.element == other.element
                and self.length == other.length)

    @property
    def full_shape(self):
        return (len(self),) + getattr(self.element, 'full_shape', ())
    @property
    def base(self):
        return getattr(self.element, 'base', self.element)
    @property
    def machine_type(self):
        return dtype((self.element.machine_type, len(self)))

class _StructMember(?):
    ...

class StructMeta(type):
    # Have a new StructMember (meta)class?
    def __new__(cls, name, *contents):
        return super().__new__(cls, name, (), {})
    def __init__(self, name, *contents):
        self.name = name
        self.contents = OrderedDict((c.name, type(c)('.'.join((name, c.name))))
                                    for c in contents)

    def __str__(self):
        return "struct {} {{ {}; }}".format(self.name, '; '.join(map(str, self)))
    def __iter__(self):
        return  iter(self.contents.values())
    def __getitem__(self, name):
        return self.contents[name]
    def __len__(self):
        return len(self.contents)

Float = _ScalarMeta('Float', dtype('float32'), '')
Double = _ScalarMeta('Double', dtype('float64'), 'd')
Int = _ScalarMeta('Int', dtype('int32'), 'i')
Uint = _ScalarMeta('Uint', dtype('uint32'), 'u')
Bool = _ScalarMeta('Bool', dtype('uint32'), 'b')

scalar_types = (Float, Double, Int, Uint, Bool)

vector_sizes = range(2, 5)
for scalar_type, length in cartesian(scalar_types, vector_sizes):
    cls = _VectorMeta(scalar_type, length)
    locals()[cls.__name__] = cls

for scalar_type, columns, rows in cartesian(scalar_types, vector_sizes, vector_sizes):
    cls = _MatrixMeta(scalar_type, columns, rows)
    locals()[cls.__name__] = cls
