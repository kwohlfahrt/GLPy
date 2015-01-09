from math import acos, sqrt, pi
import numpy
from numpy.linalg import norm

from . import xform

# FIX AXIS SIZES
class ArcBall:
	def __init__(self, center=(0, 0), axes=(1, 1)):
		self.old_rotation = numpy.identity(4)
		self.points = [None, None]
		self.center = center
		self.axes = axes
	
	def surfacePoint(self, *pt):
		pt = (numpy.asarray(pt) - self.center) / self.axes

		surface_pt = numpy.zeros(3)
		surface_pt[0:2] = pt
		if norm(pt) < 1:
			surface_pt[2] = sqrt(1 - (pt ** 2).sum())
		else:
			surface_pt[0:2] = pt / norm(pt) 

		return surface_pt
	
	def startRotation(self, *pt):
		self.points[0] = self.surfacePoint(*pt)

	def updateRotation(self, *pt):
		self.points[1] = self.surfacePoint(*pt)

	def finishRotation(self):
		self.old_rotation = self.totalRotation()
		self.points = [None, None]
		
	def currentRotation(self):
		if any(a is None for a in self.points):
			theta = 0
		else:
			theta = acos(numpy.dot(*self.points))

		if theta == 0:
			# n is arbitrary as there is no rotation
			n = (1, 0, 0)
		elif theta == pi:
			# self.points are on a line, need to be orthogonal to either
			# could do branchless as input is guaranteed to be normalized.
			p = self.points[0]
			if abs(p[0]) > abs(p[1]):
				n = (-p[1], p[0], 0)
			else:
				n = (0, -p[2], p[1])
		else:
			n = numpy.cross(*self.points)

		return xform.rotateAngleAxis(theta, n)
	
	def totalRotation(self):
		return self.currentRotation().dot(self.old_rotation)
	
	@property
	def center(self):
		return self._center
	@center.setter
	def center(self, value):
		self._center = numpy.asarray(value)
	@property
	def axes(self):
		return self._axes
	@axes.setter
	def axes(self, value):
		self._axes = numpy.asarray(value)
