#!/usr/bin/env python3

from .enums import magic_number
from .opcodes import OpCode
from .util import unpackStream, iterUnpackStream
from struct import error as StructError

from .util import PartitionedGenerator

def checkOpName(*names):
    names = set(names)
    def f(opcode):
        return opcode.name in names
    return f

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

        opcodes = PartitionedGenerator(OpCode.parse_all(f))
        source = opcodes.get(checkOpName('OpSource'), default=None)
        source_extensions = list(opcodes.yield_while(checkOpName('OpSourceExtension')))
        compile_flags = list(opcodes.yield_while(checkOpName('OpCompileFlag')))
        extensions = list(opcodes.yield_while(checkOpName('OpExtension')))
        instruction_imports = list(opcodes.yield_while(checkOpName('OpExtInstImport')))
        memory_model = opcodes.get(checkOpName('OpMemoryModel'))
        entry_points = list(opcodes.yield_while(checkOpName('OpEntryPoint')))
        execution_modes = list(opcodes.yield_while(checkOpName('OpExecutionMode')))
        strings = list(opcodes.yield_while(checkOpName('OpString')))
        names = list(opcodes.yield_while(checkOpName('OpName', 'OpMemberName')))
        lines = list(opcodes.yield_while(checkOpName('OpLine')))
        decorations = list(opcodes.yield_while(checkOpName('OpDecorate', 'OpMemberDecorate',
                                                           'OpGroupDecorate', 'OpGroupMemberDecorate',
                                                           'OpDecorationGroup')))
        # TODO: Test variable scope != `StorageClass.Function`
        types, consts, variables = map(list,
                                       opcodes.yield_any(lambda oc: oc.name.startswith('OpType'),
                                                         lambda oc: oc.name.startswith('OpConstant'),
                                                         checkOpName('OpVariable')))

        return cls(version, generator, id_bound,
                   memory_model, entry_points, source, source_extensions,
                   execution_modes, compile_flags, extensions,
                   instruction_imports, strings, names, lines, decorations,
                   types, variables, consts)# function_declarations, function_definitions)
