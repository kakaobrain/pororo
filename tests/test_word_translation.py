"""Test Word Translation module"""

import unittest

from pororo import Pororo


class PororoWordTranslationTester(unittest.TestCase):

    def test_modules(self):
        ko_en = Pororo(task="wt", lang="ko", tgt="en")
        ko_en_res = ko_en("와인")
        self.assertIsInstance(ko_en_res, list)


if __name__ == "__main__":
    unittest.main()
