import unittest
import numpy

from ..indexing import *

class TestIsContiguous(unittest.TestCase):
	def test_simple(self):
		shape = (10,)
		self.assertTrue(isContiguous([slice(6)], shape))
		self.assertTrue(isContiguous([3], shape))
		self.assertTrue(isContiguous([slice(3, 6)], shape))

	def test_step(self):
		shape = (10,)
		self.assertFalse(isContiguous([slice(3, 10, 2)], shape))
		self.assertTrue(isContiguous([slice(3, 10, 20)], shape))

	def test_multidim(self):
		shape = (5, 20)
		self.assertTrue(isContiguous([slice(2, 3), slice(None, None)], shape))
		self.assertTrue(isContiguous([slice(2, 4), slice(None, None)], shape))
		self.assertTrue(isContiguous([slice(2, 3), slice(10, 15)], shape))
		self.assertFalse(isContiguous([slice(2, 4), slice(10, 15)], shape))
		self.assertFalse(isContiguous([slice(2, 4), 10], shape))
		self.assertTrue(isContiguous([2, slice(None, None)], shape))
		self.assertTrue(isContiguous([2, slice(10, 15)], shape))
		
	def test_multidim_single_step(self):
		shape = (5, 20)
		self.assertFalse(isContiguous([slice(2, 4), slice(10, None, 100)], shape))
		self.assertTrue(isContiguous([slice(2, 3), slice(10, None, 100)], shape))
		self.assertTrue(isContiguous([2, slice(10, None, 100)], shape))

class TestFlatOffset(unittest.TestCase):
	def test_simple(self):
		shape = (10,)
		idx = (5,)
		self.assertEqual(flatOffset(idx, shape), 5)

	def test_negative(self):
		shape = (10,)
		idx = (-5,)
		self.assertEqual(flatOffset(idx, shape), 5)
		idx = (-4,)
		self.assertEqual(flatOffset(idx, shape), 6)

	def test_multiple(self):
		shape = (5, 8, 6, 4)
		idx = (0, 2, 3, 1)
		result = 1 + 3 * 4 + 2 * 6 * 4
		self.assertEqual(flatOffset(idx, shape), result)

	def test_empty(self):
		self.assertEqual(flatOffset((), ()), 0)

	def test_slice(self):
		shape = (5, 8, 6, 4)
		idx = (0, slice(2, 5), 3, slice(1, 2))
		result = 1 + 3 * 4 + 2 * 6 * 4
		self.assertEqual(flatOffset(idx, shape), result)

	def test_multidim_short(self):
		shape = (5, 8, 6, 4)
		idx = (0, 2)
		result = 2 * 6 * 4
		self.assertEqual(flatOffset(idx, shape), result)
