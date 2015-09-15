from struct import unpack, calcsize

def unpackStream(fmt, stream):
    return unpack(fmt, stream.read(calcsize(fmt)))

def iterUnpackStream(fmt, stream):
    data = stream.read(calcsize(fmt))
    while data:
        yield unpack(fmt, data)
        data = stream.read(calcsize(fmt))
