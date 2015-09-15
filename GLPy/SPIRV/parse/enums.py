from .specification import spec_tree, text, normalizeWhitespace
from .util import unpackStream
from functools import partial
from enum import Enum

class SPIRVEnumValue(int):
    def __new__(cls, word, description, required_capabilities, *operands):
        self = super().__new__(cls, word)
        self.description = description
        self.required_capabilities = required_capabilities
        self.operands = operands
        return self

class SPIRVEnum(Enum):
    # FIXME: Something weird going on with the two classmethods
    # `fromTable` is called as SPIRVEnum.fromTable(*args, **kwargs)
    # `parse` is called as SPIRVEnum(*args, **kwargs).parse(*args, **kwargs)
    @classmethod
    def fromTable(cls, table):
        enum_name = normalizeWhitespace(*table.xpath('thead/tr[1]/th[1]//text()'))
        enum_name = ''.join(enum_name.split())
        rows = ((elem.xpath('p//text()') for elem in row.xpath('td'))
                for row in table.xpath('tbody/tr'))
        values = []
        for (word,), (name, *description), *extras in rows:
            word = int(word, base=0)
            description = ''.join(description).strip()
            try:
                required_capabilities, *operands = extras
            except ValueError:
                required_capabilities, operands = [], []
            try:
                operands = [(name, normalizeWhitespace(*description)) for name, *description
                            in operands]
            except ValueError:
                operands = []
            value = SPIRVEnumValue(word, description, required_capabilities, *operands)
            values.append((name, value))
        return cls(enum_name, values)

    @classmethod
    def parse(cls, b):
        return cls(*unpackStream('I', b))

enum_names = ['Source Language',
              'Execution Model',
              'Addressing Model',
              'Memory Model',
              'Execution Mode',
              'Storage Class',
              'Dim',
              'Sampler Addressing Mode',
              'Sampler Filter Mode',
              'Image Format',
              'Image Channel Order',
              'Image Channel Data Type',
              'Image Operands',
              'FP Fast Math Mode',
              'FP Rounding Mode',
              'Linkage Type',
              'Access Qualifier',
              'Function Parameter Attribute',
              'Decoration',
              'BuiltIn',
              'Selection Control',
              'Loop Control',
              'Function Control',
              'Memory Semantics <id>',
              'Memory Access',
              'Scope <id>',
              'Group Operation',
              'Kernel Enqueue Flags',
              'Kernel Profiling Info',]

def _getTable(tree, header_id):
    table, = tree.xpath('//a[@id="{}"]/../../table'.format(header_id))
    return table

for enum in map(SPIRVEnum.fromTable, map(partial(_getTable, spec_tree), enum_names)):
    locals()[enum.__name__] = enum

magic_number = int(text(*_getTable(spec_tree, "Magic").xpath('tbody')), 16)
