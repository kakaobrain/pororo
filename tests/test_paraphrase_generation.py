"""Test Paraphrase Generation module"""

import unittest

from pororo import Pororo


class PororoParaphraseTester(unittest.TestCase):

    def test_modules(self):
        para_ko = Pororo(task="pg", lang="ko")
        para_ko_res = para_ko("나는 여기에 산다.")
        self.assertIsInstance(para_ko_res, str)

        para_en = Pororo(task="pg", lang="en")
        para_en_res = para_en("I'm good, but thanks for the offer.")
        self.assertIsInstance(para_en_res, str)

        para_ja = Pororo(task="pg", lang="ja")
        para_ja_res = para_ja("雨の日を聞く良い音楽をお勧めしてくれ。")
        self.assertIsInstance(para_ja_res, str)

        para_zh = Pororo(task="pg", lang="zh")
        para_zh_res = para_zh("我喜欢足球")
        self.assertIsInstance(para_zh_res, str)


if __name__ == "__main__":
    unittest.main()
