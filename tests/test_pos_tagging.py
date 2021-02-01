"""Test Part-Of-Speech Tagging module"""

import unittest

from pororo import Pororo


class PororoPosTester(unittest.TestCase):

    def test_modules(self):
        tagger = Pororo(task="pos", lang="ko")
        tagger_res = tagger("나는 여기에 산다.")
        self.assertIsInstance(tagger_res, list)

        nltk_tagger = Pororo(task="pos", lang="en")
        nltk_res = nltk_tagger(
            "The striped bats are hanging, on their feet for best.")
        self.assertIsInstance(nltk_res, list)


if __name__ == "__main__":
    unittest.main()
