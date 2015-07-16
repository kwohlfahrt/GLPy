#!/usr/bin/env python3

from .enums import magic_number
from .opcodes import OpCode
from .util import unpackStream, iterUnpackStream
from struct import error as StructError

def parse(f):
    magic, version, generator, id_bound, zero = unpackStream('5I', f)
    if magic != magic_number:
        raise ValueError("First word of file is 0x{:x}, not magic number 0x{:x}"
                            .format(magic, magic_number))
    if zero != 0:
        raise ValueError("Expected fifth word of file to be 0, got {}".format(zero))

    while True:
        try:
            yield OpCode.parse(f)
        except StructError:
            break
