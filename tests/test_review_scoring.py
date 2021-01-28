"""Test Review Scoring module"""

import unittest

from pororo import Pororo


class PororoReviewScoreTester(unittest.TestCase):

    def test_modules(self):
        scorer = Pororo(task="review", lang="ko")
        scorer_res = scorer("그냥저냥 다른데랑 똑같숩니다")
        self.assertIsInstance(scorer_res, float)


if __name__ == "__main__":
    unittest.main()
