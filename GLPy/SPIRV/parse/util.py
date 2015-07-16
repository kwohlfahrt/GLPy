from struct import unpack, calcsize

def unpackStream(fmt, stream):
    return unpack(fmt, stream.read(calcsize(fmt)))

def iterUnpackStream(fmt, stream):
    while True:
        data = stream.read(calcsize(fmt))
        if not data:
            break
        yield unpack(fmt, data)
