"""Test heuristic library related pipeline modules"""

import unittest

from pororo.utils import postprocess_span


class PororoUtilFuncTester(unittest.TestCase):

    def test_modules(self):
        res = postprocess_span("이민자들은")
        self.assertIsInstance(res, str)
        self.assertEqual(res, "이민자들")

        res = postprocess_span("8100억원에")
        self.assertIsInstance(res, str)
        self.assertEqual(res, "8100억원")

        res = postprocess_span("1960년대부터")
        self.assertIsInstance(res, str)
        self.assertEqual(res, "1960년대")

        res = postprocess_span("군사 목적으로는,")
        self.assertIsInstance(res, str)
        self.assertEqual(res, "군사 목적")


if __name__ == "__main__":
    unittest.main()
