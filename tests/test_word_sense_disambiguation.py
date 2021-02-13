"""Test Word Sense Disambiguation module"""

import unittest

from pororo import Pororo


class PororoWsdTester(unittest.TestCase):

    def test_modules(self):
        wsd = Pororo(task="wsd", lang="ko")
        wsd_res = wsd("머리에 이가 있나봐.")
        self.assertIsInstance(wsd_res, list)


if __name__ == "__main__":
    unittest.main()
