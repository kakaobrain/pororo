"""Test Semantic Textual Similarity module"""

import unittest

from pororo import Pororo


class PororoStsTester(unittest.TestCase):

    def test_modules(self):
        sim = Pororo(
            task="similarity",
            lang="ko",
            model="brainsbert.base.ko.kornli.korsts",
        )
        sim_res = sim("야 너 몇 살이야?", "당신의 나이는 어떻게 되십니까?")
        self.assertIsInstance(sim_res, float)

        sim = Pororo(
            task="similarity",
            lang="ko",
            model="brainbert.base.ko.korsts",
        )
        sim_res = sim("야 너 몇 살이야?", "당신의 나이는 어떻게 되십니까?")
        self.assertIsInstance(sim_res, float)


if __name__ == "__main__":
    unittest.main()
