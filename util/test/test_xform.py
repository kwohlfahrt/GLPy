from ..xform import *
import numpy

import unittest

from math import pi, radians

class XformTest(unittest.TestCase):
	def setUp(self):
		self.origin = numpy.array([0,0,0,1])
		self.unit = numpy.array([1,1,1,1])
		self.x = numpy.array([1,0,0,1])
		self.seq = numpy.array([1,2,3,1])
		
class TestTranslate(XformTest):
	def test_default_no_trans(self):
		for i in (self.origin, self.unit, self.seq):
			self.assertTrue((translate().dot(i) == i).all())
	
	def test_single_axis(self):
		for i in (self.origin, self.unit, self.seq):
			expect = i.copy()
			expect[0] += 1
			self.assertTrue(numpy.allclose(translate(x=1).dot(i), expect))
			expect = i.copy()
			expect[1] += 1
			self.assertTrue(numpy.allclose(translate(y=1).dot(i), expect))
	
	def test_multiple_axis(self):
		for i in (self.origin, self.unit, self.seq):
			expect = i.copy()
			expect[1:3] += 1
			self.assertTrue(numpy.allclose(translate(y=1, z=1).dot(i), expect))

class TestRotate(XformTest):
	def test_default_no_rot(self):
		for i in self.origin, self.unit, self.seq:
			self.assertTrue((rotate().dot(i) == i).all())
	
	def test_nop(self):
		self.assertTrue((rotate(x=1).dot(self.x) == self.x).all())
		self.assertTrue((rotate(x=1, z=1).dot(self.origin) == self.origin).all())
		self.assertTrue(numpy.allclose(rotate(y=2 * pi).dot(self.x), self.x))

	def test_axis(self):
		y = numpy.array([0, 1, 0, 1])
		z = numpy.array([0, 0, 1, 1])

		self.assertTrue(numpy.allclose(rotate(y=pi/2).dot(self.x), z))
		self.assertTrue(numpy.allclose(rotate(z=pi/2).dot(self.x), y))

class TestRotateAngleAxis(XformTest):
	def test_nop(self):
		for i in self.origin, self.unit, self.seq:
			self.assertTrue((rotateAngleAxis(0, (1, 0, 0)).dot(i) == i).all())

class TestScale(XformTest):
	def test_default_no_scale(self):
		for i in (self.origin, self.unit, self.seq):
			self.assertTrue((rotate().dot(i) == i).all())
	
	def test_nop(self):
		self.assertTrue((scale(y=2, z=3).dot(self.x) == self.x).all())
		self.assertTrue((scale(y=2, z=3).dot(self.origin) == self.origin).all())
	
	def test_uniform(self):
		expect = numpy.array([2, 0, 0, 1])
		self.assertTrue(numpy.allclose(scale(u=2).dot(self.x), expect))
		expect = numpy.array([2, 2, 2, 1])
		self.assertTrue(numpy.allclose(scale(u=2).dot(self.unit), expect))
		expect = numpy.array([2, 4, 6, 1])
		self.assertTrue(numpy.allclose(scale(u=2).dot(self.seq), expect))
	
	def test_simple(self):
		expect = numpy.array([1, 3, 3, 1])
		self.assertTrue(numpy.allclose(scale(y=1.5).dot(self.seq), expect))

class TestLookAt(XformTest):
	def test_zero_position_error(self):
		self.assertRaises(ZeroDivisionError, lookAt, self.origin[:3])
		self.assertRaises(ZeroDivisionError, lookAt, self.seq[:3], self.seq[:3])

	def test_parallel_up_error(self):
		self.assertRaises(ZeroDivisionError, lookAt, self.x[:3], up=self.x[:3])
		self.assertRaises(ZeroDivisionError, lookAt, self.seq[:3], up=self.seq[:3])
	
	def test_up_insensitive(self):
		self.assertTrue(numpy.allclose(lookAt((0, 0, 1)), lookAt((0, 0, 1), up=(0, 0.1, 0))))
		self.assertTrue(numpy.allclose(lookAt((0, 0, 1)), lookAt((0, 0, 1), up=(0, 0.1, -1))))
		self.assertTrue(numpy.allclose(lookAt((0, 0, 1)), lookAt((0, 0, 1), up=(0, 0.1, 1))))
	
	def test_z(self):
		expect = numpy.array([
			[1,0,0,0],
			[0,1,0,0],
			[0,0,1,0],
			[0,0,0,1],
		])
		self.assertTrue(numpy.allclose(lookAt(self.origin[:3], (0, 0, -1)), expect))
		expect = numpy.array([
			[1,0,0, 0],
			[0,1,0, 0],
			[0,0,1,-1],
			[0,0,0, 1],
		])
		self.assertTrue(numpy.allclose(lookAt((0, 0, 1), self.origin[:3]), expect))
	
	def test_x(self):
		expect = rotate(y=pi/2).dot(translate(x=-1))
		self.assertTrue(numpy.allclose(lookAt(self.x[:3]), expect))
	
	def test_y(self):
		expect = rotate(x=pi/2).dot(translate(y=-1))
		self.assertTrue(numpy.allclose(lookAt((0, 1, 0), up=(0, 0, -1)), expect))

class TestViewAxis(unittest.TestCase):
	def setUp(self):
		self.cam = lookAt((0, 0, 1))
		self.persp_cam = perspective().dot(self.cam)

	def test_default(self):
		self.assertEqual(viewAxis(self.cam.dot(numpy.identity(4))), 2)
		self.assertEqual(viewAxis(self.persp_cam.dot(numpy.identity(4))), 2)
	
	def test_rotate(self):
		self.assertEqual(viewAxis(self.cam.dot(rotate(y=pi/2))), 0)
		self.assertEqual(viewAxis(self.cam.dot(rotate(y=pi))), 2)
		self.assertEqual(viewAxis(self.cam.dot(rotate(x=pi/2))), 1)

	def test_rotate_persp(self):
		self.assertEqual(viewAxis(self.persp_cam.dot(rotate(y=pi/2))), 0)
		self.assertEqual(viewAxis(self.persp_cam.dot(rotate(y=pi))), 2)
		self.assertEqual(viewAxis(self.persp_cam.dot(rotate(x=pi/2))), 1)
	
	def test_translate(self):
		self.assertEqual(viewAxis(self.cam.dot(translate(x=100))), 2)
		self.assertEqual(viewAxis(self.cam.dot(translate(y=100))), 2)
		self.assertEqual(viewAxis(self.cam.dot(translate(z=100))), 2)

	def test_translate_persp(self):
		self.assertEqual(viewAxis(self.persp_cam.dot(translate(x=100))), 0)
		self.assertEqual(viewAxis(self.persp_cam.dot(translate(y=100))), 1)
		self.assertEqual(viewAxis(self.persp_cam.dot(translate(z=100))), 2)
	
	def test_scale_rotate(self):
		self.assertEqual(viewAxis(self.cam.dot(rotate())), 2)
		self.assertEqual(viewAxis(self.cam.dot(rotate(y=radians(46)))), 0)
		self.assertEqual(viewAxis(self.cam.dot(rotate(y=radians(46))).dot(scale(x=10))), 2)
	
	def test_scale_translate(self):
		self.assertEqual(viewAxis(self.persp_cam.dot(translate())), 2)
		self.assertEqual(viewAxis(self.persp_cam.dot(translate(x=1.1))), 0)
		self.assertEqual(viewAxis(self.persp_cam.dot(translate(x=1.1)).dot(scale(x=10))), 2)
