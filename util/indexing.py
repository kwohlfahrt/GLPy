from itertools import accumulate, chain, repeat
import operator

def isContiguous(idxs, shape):
	idxs = (range(*i.indices(s)) if isinstance(i, slice) else range(i, i+1)
	        for i, s in zip(idxs, shape))
	shape = iter(shape)
	for i, s in zip(idxs, shape):
		if len(i) != 1:
			if i.step != 1:
				return False
			break
	if any(i != range(s) for i, s in zip(idxs, shape)):
		return False
	return True

def flatOffset(idxs, shape, base=1):
	idxs = chain(idxs, repeat(0, len(shape) - len(idxs)))
	idxs = [idx.indices(s)[0] if isinstance(idx, slice) else idx % s
	        for idx, s in zip(idxs, shape)]
	shape = chain((base,), reversed(shape))
	shapes = accumulate(shape, operator.mul)
	return sum( idx * s for idx, s in zip(reversed(idxs), shapes) )
