from struct import unpack, calcsize
from itertools import count

def unpackStream(fmt, stream):
    return unpack(fmt, stream.read(calcsize(fmt)))

def iterUnpackStream(fmt, stream):
    data = stream.read(calcsize(fmt))
    while data:
        yield unpack(fmt, data)
        data = stream.read(calcsize(fmt))

def getBits(number):
    for bit in count():
        test = 0x1 << bit
        if test & number:
            yield bit
        if test > number:
            break
