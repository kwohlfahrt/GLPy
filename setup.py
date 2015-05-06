#!/usr/bin/env python3

'''
	GLPy
	----

	A set of python classes built on PyOpenGL to make it easier to create OpenGL graphics.
'''

from setuptools import setup

setup(
	name='GLPy',
	version='0.1',
	author='Kai Wohlfahrt',
	description='Pythonic OpenGL classes',
	long_description=__doc__,
	packages=['GLPy'],
	platforms='any',
	install_requires=[],
	test_suite='test'
)


