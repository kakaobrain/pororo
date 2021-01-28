"""Test Lemmatization module"""

import unittest

from pororo import Pororo


class PororoLemmaTester(unittest.TestCase):

    def test_modules(self):
        lemma = Pororo(task="lemma", lang="en")
        lemma_res = lemma("He loves me.")
        self.assertIsInstance(lemma_res, list)


if __name__ == "__main__":
    unittest.main()
