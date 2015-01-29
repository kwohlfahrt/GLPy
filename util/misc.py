from itertools import count, accumulate, chain
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

def isIterable(i):
	try:
		iter(i)
	except TypeError:
		return False
	else:
		return True

def equal(iterable):
	iterable = iter(iterable)
	r = next(iterable)
	return all(i == r for i in iterable)

def contains(subsequence, sequence):
	# Could optimize, see Boyer-Moore
	n = len(subsequence)
	return any(sequence[i:i+n] == subsequence for i in range(len(sequence) - n + 1))
