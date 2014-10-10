from itertools import count
from functools import reduce
import operator

def product(l, acc=1):
	return reduce(operator.mul, l, acc)

def last(iterable):
	iterable = iter(iterable)
	r = next(iterable)
	for r in iterable:
		pass
	return r

def subIter(subscriptable):
	for i in count():
		try:
			yield subscriptable[i]
		except IndexError:
			raise StopIteration

def isContiguous(idxs, shape):
	idxs = (range(*i.indices(s)) for i, s in zip(idxs, shape))
	shape = iter(shape)
	for i, s in zip(idxs, shape):
		if i.step != 1 and len(i) != 1:
			return False
		if len(i) != 1:
			break
	if any(i != range(s) for i, s in zip(idxs, shape)):
		return False
	return True
