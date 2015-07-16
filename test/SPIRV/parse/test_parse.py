from GLPy.SPIRV import parse
import unittest
from os import path

class TestParse(unittest.TestCase):
    def test_runs(self):
        spirv_path = path.join(path.dirname(__file__), 'example.frag.spv')
        with open(spirv_path, 'rb') as f:
            ops = list(parse(f))
        self.assertEqual(len(ops), 128)
