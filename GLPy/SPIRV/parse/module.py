#!/usr/bin/env python3

from .enums import magic_number
from .opcodes import *
from .util import unpackStream, iterUnpackStream
from struct import error as StructError

def parseAll(f, *opcodes):
    while True:
        for opcode in opcodes:
            try:
                r = opcode.parse(f)
            except ValueError:
                pass
            else:
                break
        else:
            break
        yield r

class Module:
    def __init__(self, version, generator_id, id_bound,
                 memory_model, entry_points, source=None,
                 source_extensions=(), execution_modes=(), compile_flags=(),
                 extensions=(), instruction_imports=(),
                 strings=(), names=(), lines=(), decorations=(),
                 types=(), global_variables=(), constants=(),
                 function_declarations=(), function_definitions=()):
        self.version = version
        self.generator_id = generator_id
        self.id_bound = id_bound

        self.memory_model = memory_model
        self.entry_points = entry_points # May have 0 if using `Link`
        self.execution_modes = execution_modes
        self.compile_flags = compile_flags
        self.extensions = extensions
        self.instruction_imports = instruction_imports
        self.strings = strings
        self.names = names
        self.lines = lines
        self.decorations = decorations
        self.types = types
        self.global_variables = global_variables
        self.constants = constants
        self.function_declarations = function_declarations
        self.function_definitions = function_definitions

    @classmethod
    def fromFile(cls, f):
        magic, version, generator, id_bound, zero = unpackStream('5I', f)
        if magic != magic_number:
            raise ValueError("First word of file is 0x{:x}, not magic number 0x{:x}"
                                .format(magic, magic_number))
        if zero != 0:
            raise ValueError("Expected fifth word of file to be 0, got {}".format(zero))

        source = OpSource.parse(f)
        source_extensions = list(parseAll(f, OpSourceExtension))
        compile_flags = list(parseAll(f, OpCompileFlag))
        extensions = list(parseAll(f, OpExtension))
        instruction_imports = list(parseAll(f, OpExtInstImport)) # Has result ID
        memory_model = OpMemoryModel.parse(f)
        entry_points = list(parseAll(f, OpEntryPoint))
        execution_modes = list(parseAll(f, OpExecutionMode))
        strings = list(parseAll(f, OpString)) # Has result ID
        names = list(parseAll(f, OpName, OpMemberName)) # May forward reference
        lines = list(parseAll(f, OpLine)) # May forward reference
        decorations = list(parseAll(f, OpDecorate, OpMemberDecorate,
                                    OpGroupDecorate, OpGroupMemberDecorate,
                                    OpDecorationGroup)) # May forward reference
        tcv = list(parseAll(f, OpVariable, *(constant_opcodes + type_opcodes)))
        types = list(filter(lambda o: o.name.startswith('OpType'), tcv))
        variables = list(filter(lambda o: o is OpVariable, tcv))
        consts = list(filter(lambda o: o.name.startswith('OpConstant'), tcv))

        return cls(version, generator, id_bound,
                   memory_model, entry_points, source, source_extensions,
                   execution_modes, compile_flags, extensions,
                   instruction_imports, strings, names, lines, decorations,
                   types, variables, consts, # function_declarations, function_definitions)
                   )
