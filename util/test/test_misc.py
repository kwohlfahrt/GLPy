#!/usr/bin/env python3

import unittest
from ..misc import *

import itertools
import numpy

class TestProduct(unittest.TestCase):
	def setUp(self):
		self.seq = [1, 2, 3]
		self.seq_prod = 6

	def test_empty(self):
		self.assertEqual(product([]), 1)

	def test_zero(self):
		self.assertEqual(product([0]), 0)
		self.assertEqual(product([0, 5]), 0)

	def test_ones(self):
		for i in range(3):
			self.assertEqual(product([1] * i), 1)

	def test_sequence(self):
		self.assertEqual(product(self.seq), self.seq_prod)

	def test_generator(self):
		g = (i for i in range(3))
		self.assertEqual(product(g), 0)
		g = (i for i in self.seq)
		self.assertEqual(product(g), self.seq_prod)

class TestLast(unittest.TestCase):
	def test_basic(self):
		l = [1, 54, 12, 10, 20]
		self.assertEqual(last(l), l[-1])

	def test_error_empty(self):
		self.assertRaises(StopIteration, last, [])

class TestSubIter(unittest.TestCase):
	def test_list(self):
		l = [1, 2, 4, 1]
		self.assertEqual(list(subIter(l)), l)

class TestEqual(unittest.TestCase):
	def test_equal(self):
		g = itertools.repeat(1, 5)
		self.assertTrue(equal(g))

	def test_not_equal(self):
		g = range(5)
		self.assertFalse(equal(g))

class TestContains(unittest.TestCase):
	def test_simple(self):
		self.assertTrue(contains([1, 2], [1, 2, 3, 4, 5]))
		self.assertTrue(contains([3, 4], [1, 2, 3, 4, 5]))
		self.assertTrue(contains([4, 5], [1, 2, 3, 4, 5]))
		self.assertFalse(contains([5, 4], [1, 2, 3, 4, 5]))
		self.assertFalse(contains([1, 3], [1, 2, 3, 4, 5]))
