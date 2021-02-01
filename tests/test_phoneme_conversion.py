"""Test Grapheme to Phoneme module"""

import unittest

from pororo import Pororo


class PororoPhonemeConversionTester(unittest.TestCase):

    def test_modules(self):
        g2pk = Pororo(task="g2p", lang="ko")
        g2pk_res = g2pk("어제는 날씨가 맑았는데, 오늘은 흐리다.")
        self.assertIsInstance(g2pk_res, str)

        g2pen = Pororo(task="g2p", lang="en")
        g2pen_res = g2pen("I have $250 in my pocket.")
        self.assertIsInstance(g2pen_res, list)

        g2pzh = Pororo(task="g2p", lang="zh")
        g2pzh_res = g2pzh("然而，他红了20年以后，他竟退出了大家的视线。")
        self.assertIsInstance(g2pzh_res, str)


if __name__ == "__main__":
    unittest.main()
