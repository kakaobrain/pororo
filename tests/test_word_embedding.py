"""Test Word Embedding module"""

import unittest
from collections import OrderedDict

from pororo import Pororo


class PororoWordEmbeddingTester(unittest.TestCase):

    def test_modules(self):
        wv = Pororo(task="wordvec", lang="ko")
        wv_res = wv("와인")
        self.assertIsInstance(wv_res, OrderedDict)


if __name__ == "__main__":
    unittest.main()
