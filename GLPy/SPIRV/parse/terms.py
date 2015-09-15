from struct import error as StructError

from typing import List, NamedTuple

class LiteralNumber(int):
    @classmethod
    def parse(cls, b):
        #FIXME: Can be > 1 word long
        return cls(*unpackStream('I', b))

class LiteralString(str):
    @classmethod
    def parse(cls, b):
        return cls(b.read().decode('UTF-8'))

class Id(int):
    @classmethod
    def parse(cls, b):
        return cls(*unpackStream('I', b))

    def __repr__(self):
        return "<{}>".format(super().__str__())

class ResultId(Id):
    pass

class OptionalId(Id):
    @classmethod
    def parse(cls, b):
        try:
            return cls(*unpackStream('I', b))
        except StructError:
            return None

class ImageOperands:
    def __init__(self, flags, operands):
        self.flags = set()
        self.operands = operands

    @classmethod
    def parse(cls, b):
        flags = unpackStream('I', b)
        operands = MultiLiteral.parse(b)
        return cls(flags, operands)

# TODO: Below are classes that could be combined in a re-factor
# A general 'pair', and a general 'multi_x'

# Cannot subclass regular tuple
IdLiteralTuple = NamedTuple('IdLiteralTuple', [('id', Id), ('literal', LiteralNumber)])

class IdLiteralPair(IdLiteralTuple):
    def __new__(cls, id, literal):
        return super().__new__(cls, Id(id), LiteralNumber(literal))

    @classmethod
    def parse(cls, b):
        return cls(*unpackStream('II', b))

# Cannot subclass regular tuple
LiteralIdTuple = NamedTuple('LiteralIdTuple', [('literal', LiteralNumber), ('id', Id)])

class LiteralIdPair(LiteralIdTuple):
    def __new__(cls, literal, id):
        return super().__new__(cls, LiteralNumber(literal), Id(id))

    @classmethod
    def parse(cls, b):
        return cls(*unpackStream('II', b))


class MultiId(List[Id]):
    def __init__(self, *ids):
        super().__init__(map(Id, ids))

    @classmethod
    def parse(cls, b):
        return cls(*iterUnpackStream('I', b))

class MultiLiteral(List[LiteralNumber]):
    def __init__(self, *ids):
        super().__init__(map(LiteralNumber, ids))

    @classmethod
    def parse(cls, b):
        return cls(*iterUnpackStream('I', b))

class MultiIdLiteralPair(List[IdLiteralPair]):
    def __init__(self, *pairs):
        super().__init__(map(IdLiteralPair, pairs))

    @classmethod
    def parse(cls, b):
        return cls(*iterUnpackStream('II', b))

class MultiLiteralIdPair(List[LiteralIdPair]):
    def __init__(self, *pairs):
        super().__init__(map(LiteralIdPair, pairs))

    @classmethod
    def parse(cls, b):
        return cls(*iterUnpackStream('II', b))
