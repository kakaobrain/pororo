"""Test Morphological Inflection module"""

import unittest

from pororo import Pororo


class PororoInflectionTester(unittest.TestCase):

    def test_modules(self):
        inflect = Pororo(task="inflection", lang="ko")
        inflect_res = inflect("곱")
        self.assertIsInstance(inflect_res, list)


if __name__ == "__main__":
    unittest.main()
