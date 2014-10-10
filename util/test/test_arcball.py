from ..arcball import *
from .. import xform

import unittest, warnings
from copy import deepcopy
from math import radians, pi

class ArcBallTester(unittest.TestCase):
	def setUp(self):
		warnings.simplefilter("error")
		
	def tearDown(self):
		warnings.simplefilter("default")

	def test_surface_pt(self):
		ball = ArcBall()
		point = numpy.array([1, 1])
		expect = numpy.append(point, 0) / numpy.linalg.norm(point)
		self.assertTrue(numpy.allclose(ball.surfacePoint(*point), expect))
		self.assertTrue(numpy.allclose(ball.surfacePoint(1, 0), numpy.array([1, 0, 0])))
		self.assertTrue(numpy.allclose(ball.surfacePoint(0, 0), numpy.array([0, 0, 1])))
	
	def test_no_rotation(self):
		x = ArcBall()
		self.assertTrue((x.currentRotation() == numpy.identity(4)).all())
		self.assertTrue((x.totalRotation() == numpy.identity(4)).all())

	def test_zero_rotation(self):
		x = ArcBall()
		x.startRotation(0, 0)
		x.updateRotation(0, 0)
		self.assertTrue((x.currentRotation() == numpy.identity(4)).all())
		self.assertTrue((x.totalRotation() == numpy.identity(4)).all())
		x.finishRotation()
		self.assertTrue((x.currentRotation() == numpy.identity(4)).all())
		self.assertTrue((x.totalRotation() == numpy.identity(4)).all())
	
	def test_y_90_rotation(self):
		ball = ArcBall()
		ball.startRotation(1, 0)
		ball.updateRotation(0, 0)
		self.assertTrue((ball.totalRotation() == xform.rotate(y=pi/2)).all())
		
	def test_y_180_rotation(self):
		ball = ArcBall()
		ball.startRotation(1, 0)
		ball.updateRotation(-1, 0)
		self.assertTrue(numpy.allclose(ball.totalRotation(), xform.rotate(y=pi)))
	
	def test_segmented_rotation(self):
		ball = ArcBall()
		ball.startRotation(1, 0)

		ball2 = deepcopy(ball)
		ball2.updateRotation(0.6, -0.4)
		ball2.updateRotation(-1, 0)

		ball.updateRotation(-1, 0)
		self.assertTrue((ball.totalRotation() == ball2.totalRotation()).all())
	
	def test_finish_rotation(self):
		ball = ArcBall()
		self.assertTrue((ball.old_rotation == numpy.identity(4)).all())
		self.assertTrue((ball.currentRotation() == numpy.identity(4)).all())
		ball.startRotation(0.5, 0.6)
		self.assertTrue((ball.old_rotation == numpy.identity(4)).all())
		self.assertTrue((ball.currentRotation() == numpy.identity(4)).all())
		ball.updateRotation(0.7, -0.8)
		self.assertTrue((ball.old_rotation == numpy.identity(4)).all())
		self.assertFalse((ball.currentRotation() == numpy.identity(4)).all())
		r = ball.currentRotation()
		ball.finishRotation()
		self.assertFalse((ball.old_rotation == numpy.identity(4)).all())
		self.assertTrue((ball.old_rotation == r).all())
		self.assertTrue((ball.currentRotation() == numpy.identity(4)).all())
	
	def test_composite_rotation(self):
		ball = ArcBall()
		ball.startRotation(1, 0)
		ball.updateRotation(0, 0)
		ball.finishRotation()
		self.assertTrue((ball.totalRotation() == xform.rotate(y=pi/2)).all())
		ball.startRotation(0, 0)
		ball.updateRotation(0, -1)
		ball.finishRotation()
		self.assertTrue(numpy.allclose(ball.totalRotation(), xform.rotate(y=pi/2, x=pi/2)))
