"""Test Grapheme-to-phoneme module"""

import unittest

from pororo import Pororo


class PororoGraphemeConversionTester(unittest.TestCase):

    def test_modules(self):
        p2g_zh = Pororo(task="p2g", lang="zh")
        p2g_zh_res = p2g_zh([
            "ran2",
            "er2",
            "，",
            "ta1",
            "hong2",
            "le5",
            "20",
            "nian2",
            "yi3",
            "hou4",
            "，",
            "ta1",
            "jing4",
            "tui4",
            "chu1",
            "le5",
            "da4",
            "jia1",
            "de5",
            "shi4",
            "xian4",
            "。",
        ])
        self.assertIsInstance(p2g_zh_res, list)

        p2g_ja = Pororo(task="p2g", lang="ja")
        p2g_ja_res = p2g_ja("python ga daisuki desu。")
        self.assertIsInstance(p2g_ja_res, str)


if __name__ == "__main__":
    unittest.main()
