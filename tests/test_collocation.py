"""Test Collocation module"""

import unittest

from pororo import Pororo


class PororoCollocationTester(unittest.TestCase):

    def test_modules(self):
        col = Pororo(task="collocate", lang="ko")
        col_res = col("곱")
        self.assertIsInstance(col_res, dict)

        col_res = col("먹")
        self.assertIsInstance(col_res, dict)

        col_res = col("습")
        self.assertIsInstance(col_res, dict)

        col_res = col("덥")
        self.assertIsInstance(col_res, dict)


if __name__ == "__main__":
    unittest.main()
