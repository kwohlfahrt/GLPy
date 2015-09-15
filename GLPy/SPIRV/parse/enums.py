from .specification import spec_tree, text, normalizeWhitespace, text
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

def _getEnumTables(tree):
    binary_section_xpath = '//a[@id="Binary"]/../../div[@class="sectionbody"]'

    exclusion_template = 'not(descendant::a[@id="{}"])'
    exclusions = ' and '.join(exclusion_template.format(e) for e in ("Instructions", "Magic"))

    enum_xpath = '/div[@class="sect2" and {}]'.format(exclusions)
    return tree.xpath(binary_section_xpath + enum_xpath + '/table')

for enum in map(SPIRVEnum.fromTable, _getEnumTables(spec_tree)):
    locals()[enum.__name__] = enum

magic_number = int(text(spec_tree.xpath('//a[@id="Magic"]/../../table/tbody')[0]), 16)
