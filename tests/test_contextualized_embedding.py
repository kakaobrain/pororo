"""Test Contextualized Embedding module"""

import unittest

import numpy as np

from pororo import Pororo


class PororoContextualizedTester(unittest.TestCase):

    def test_modules(self):
        cse = Pororo(task="cse", lang="ko")
        cse_res = cse("나는 동물을 좋아하는 사람이야")
        self.assertIsInstance(cse_res, np.ndarray)

        cse = Pororo(task="cse", lang="zh")
        cse_res = cse("一群人抬头看着建筑物屋顶边缘的3人。")
        self.assertIsInstance(cse_res, np.ndarray)

        cse = Pororo(task="cse", lang="ja")
        cse_res = cse("おはようございます")
        self.assertIsInstance(cse_res, np.ndarray)


if __name__ == "__main__":
    unittest.main()
