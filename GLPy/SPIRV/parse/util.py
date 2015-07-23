from struct import unpack, calcsize
from itertools import tee
from functools import partial

def unpackStream(fmt, stream):
    return unpack(fmt, stream.read(calcsize(fmt)))

def iterUnpackStream(fmt, stream):
    while True:
        data = stream.read(calcsize(fmt))
        if not data:
            break
        yield unpack(fmt, data)

NoDefault = object()

class PartitionedGenerator:
    def __init__(self, generator):
        self.g = generator
        self.n = next(self.g)

    def yield_while(self, predicate):
        while predicate(self.n):
            yield self.n
            self.n = next(self.g)

    def yield_any(self, *predicates):
        outputs = self.yield_while(lambda x: any(p(x) for p in predicates))
        outputs = tee(outputs, len(predicates))
        outputs = map(filter, predicates, outputs)
        return outputs

    def get(self, predicate, default=NoDefault):
        if predicate(self.n):
            r = self.n
            self.n = next(self.g)
            return r
        else:
            if default is NoDefault:
                raise ValueError("Next value does not match predicate.")
            else:
                return default
