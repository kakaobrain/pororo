"""Test Natural Language Inference module"""

import unittest

from pororo import Pororo


class PororoNliTester(unittest.TestCase):

    def test_modules(self):
        nli = Pororo(task="nli", lang="ko")
        nli_res = nli(
            "BrainBert는 한국어 코퍼스에 학습된 언어모델이다.",
            "BrainBert는 한국어 모델이다.",
        )
        self.assertIsInstance(nli_res, str)


if __name__ == "__main__":
    unittest.main()
