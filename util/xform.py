import numpy
from numpy.linalg import norm, inv

from math import cos, sin, tan, pi
from functools import partial

from collections import namedtuple

def translate(x=0, y=0, z=0):
	m = numpy.identity(4)
	m[:3, 3] = [x, y, z]
	return m

def scale(x=1, y=1, z=1, u=1):
	m = numpy.identity(4)
	# NumPY > 1.10 only
	# d = m.diagonal()
	# d = [x, y, z]
	m[0,0] = x * u
	m[1,1] = y * u
	m[2,2] = z * u

	return m

def rotate(x=0, y=0, z=0):
	m_x = numpy.identity(4)
	m_y = numpy.identity(4)
	m_z = numpy.identity(4)

	if x:
		m_x[1, 1] = cos(x)
		m_x[1, 2] = -sin(x)
		m_x[2, 1] = sin(x)
		m_x[2, 2] = cos(x)

	if y:
		m_y[0, 0] = cos(y)
		m_y[0, 2] = -sin(y)
		m_y[2, 0] = sin(y)
		m_y[2, 2] = cos(y)
	
	if z:
		m_z[0, 0] = cos(z)
		m_z[0, 1] = -sin(z)
		m_z[1, 0] = sin(z)
		m_z[1, 1] = cos(z)
	
	return m_x.dot(m_y).dot(m_z)

def rotateAngleAxis(theta, v):
	v = v / norm(v)
	e = numpy.array([
		[    0,-v[2], v[1]],
		[ v[2],    0,-v[0]],
		[-v[1], v[0],    0],
	])

	v = v[:, numpy.newaxis]
	m = numpy.identity(4)
	m[0:3, 0:3] = numpy.identity(3) * cos(theta)\
		+ e * sin(theta)\
		+ (1 - cos(theta)) * v.dot(v.transpose())
	return m

def perspective(fov=pi/2, near=0.01, far=10, aspect=1):
	fov = min(fov, pi)

	s = 1 / tan( fov * 0.5 )
	m = numpy.identity(4)
	m = numpy.array([
		[s / aspect, 0, 0, 0],
		[0 , s, 0, 0],
		[0 , 0, (far + near) / (near - far), 2 * far * near / (near - far)],
		[0 , 0, -1, 0]
	])
	return m

def lookAt(position, target=(0, 0, 0), up=(0, 1, 0)):
	position = numpy.array(position)
	target = numpy.array(target)
	up = numpy.array(up)

	f = target - position
	f_norm = norm(f)
	if f_norm == 0:
		raise ZeroDivisionError("The position and target ({}, {}) must not be equal.".format(position, target))
	f = f / norm(f)

	up_norm = norm(up)
	if up_norm == 0:
		raise ZeroDivisionError("The norm of up ({}) must not be 0.".format(up))
	up = up / norm(up)

	s = numpy.cross(f, up)
	s_norm = norm(s)
	if s_norm == 0:
		raise ZeroDivisionError("The up and view vectors ({},{}) must not be parallel.".format(up, f))
	s = s / s_norm
	u = numpy.cross(s, f)
	
	m = numpy.identity(4)
	m[0:3, 0:3] = numpy.array([
		s, u, -f,
	])
	t = translate(*(-position))

	return m.dot(t)

def viewAxis(xform):
	I = numpy.identity(4)
	axes = ( I[:3,i] for i in range(3) )
	origin = I[:,3]

	# for perspective transform w ∝ z, so xform[3, 2] ≠ 0
	if not xform[3, 2]:
		origin[2] = 1
		xform[:3, 3] = 0

	position = inv(xform).dot(origin)
	dots = [ abs(position[:3].dot(i)) for i in axes ]

	return dots.index(max(dots))
