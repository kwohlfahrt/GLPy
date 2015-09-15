from .enums import *
from .terms import *

from .specification import spec_tree, normalizeWhitespace, text
from .util import unpackStream

from io import BytesIO
from struct import calcsize
from os import SEEK_CUR

class OpParam:
    def __init__(self, param_type, description):
        self.description = description
        try:
            # Get from terms and enums
            self.type = globals()[''.join(param_type.split())]
        except KeyError:
            if param_type in ('<id>', 'Scope <id>', 'Memory Semantics <id>'):
                self.type = Id
            elif param_type == 'Result <id>':
                self.type = ResultId
            elif param_type == 'Optional <id>':
                self.type = OptionalId
            elif param_type == '<id>, <id>, …':
                self.type = MultiId
            elif param_type in ('literal, literal, …', 'optional literal(s)'):
                self.type = MultiLiteral
            elif param_type == '<id>, literal, <id>, literal, …':
                self.type = MultiIdLiteralPair
            elif param_type == 'literal, label <id>, literal, label <id>, …':
                self.type = MultiLiteralIdPair
            elif param_type == 'Optional Image Operands, <id>, <id>, …':
                #TODO: Implement these
                self.type = ImageOperands
            else:
                raise

    @classmethod
    def fromElement(cls, element):
        param_type = element.xpath('./p/*[count(preceding-sibling::br) < 1]//text()')
        param_type = normalizeWhitespace(*param_type)
        description = element.xpath('./p/*[count(preceding-sibling::br) >= 1]//text()')
        description = normalizeWhitespace(*description)
        return cls(param_type, description)

    def __repr__(self):
        return "<OpParam {}>".format(self.type)

    @property
    def is_result(self):
        return (self.type == ResultId)
    @property
    def is_result_type(self):
        return (self.type == Id and self.description == 'Result Type')

class OpCode:
    opcodes = {}
    def __new__(cls, name, description, required_capability,
                min_length, word, *params):
        obj = super().__new__(cls)
        cls.opcodes[word] = obj
        return obj
    def __init__(self, name, description, required_capability,
                    min_length, word, *params):
        self.name = name
        self.word = word
        self.description = description
        self.min_length = min_length
        self.params = params
        self.required_capability = required_capability

    @property
    def has_result(self):
        return any(p.is_result for p in self.params)

    @property
    def has_result_type(self):
        return any(p.is_result_type for p in self.params)

    @classmethod
    def fromTable(cls, table):
        header, params = table.xpath('tr')
        description_box, *metadata = header.xpath('td')
        name, *description = description_box.xpath('p//text()')
        description = normalizeWhitespace(*description)
        try:
            required_capability = metadata[0].xpath('.//text()')[-1]
        except IndexError:
            required_capability = None
        length, word, *params = params.xpath('td')
        min_length = int(text(length).split('+')[0].strip())
        word = int(text(word), 0)
        params = list(map(OpParam.fromElement, params))
        return cls(name, description, required_capability, min_length, word, *params)

    def __str__(self):
        return self.name
    def __repr__(self):
        return "<OpCode '{}' [{}]>".format(self.name, self.word)

    def fromBytes(self, b):
        if len(b) < (self.min_length - 1):
            raise ValueError("Invalid length {} for {}".format(len(b), self))

        b = BytesIO(b)
        params = [p.type.parse(b) for p in self.params]

        if self.has_result_type:
            prefix = "{1:>4}: {0:>4}".format(*params)
            params = params[2:]
        elif self.has_result:
            prefix = "{1:>4}: {0:>4}".format('', *params)
            params = params[1:]
        else:
            prefix = "{0:>4}  {0:>4}".format('')
        return self

    def parse(self, f):
        opcode_format = '2H'
        opcode, oplength = unpackStream(opcode_format, f)
        if opcode != self.word:
            f.seek(-calcsize(opcode_format), SEEK_CUR)
            raise ValueError("Incorrect opcode {}, expected {}"
                             .format(opcode, self.word))
        # oplength is in words, and includes (opcode, oplength)
        return self.fromBytes(f.read((oplength - 1) * 4))

opcodes = map(OpCode.fromTable,
              spec_tree.xpath('//a[@id="Instructions"]/../../div/table/tbody'))
constant_opcodes = []
type_opcodes = []
for opcode in opcodes:
    if opcode.name.startswith('OpConstant'):
        constant_opcodes.append(opcode)
    if opcode.name.startswith('OpType'):
        type_opcodes.append(opcode)
    locals()[opcode.name] = opcode
