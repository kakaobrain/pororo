"""Test Similar Word Finding module"""

import unittest
from collections import OrderedDict

from pororo import Pororo


class PororoFindWordTester(unittest.TestCase):

    def test_modules(self):
        ft = Pororo(task="wordvec", lang="ko")
        fr_res = ft.find_similar_words("사과")
        self.assertIsInstance(fr_res, OrderedDict)


if __name__ == "__main__":
    unittest.main()
