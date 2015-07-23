from GLPy.SPIRV import Module
import unittest
from os import path

class TestParse(unittest.TestCase):
    def test_runs(self):
        spirv_path = path.join(path.dirname(__file__), 'example.frag.spv')
        with open(spirv_path, 'rb') as f:
            module = Module.fromFile(f)
        self.assertEqual(len(module.global_variables), 7)
