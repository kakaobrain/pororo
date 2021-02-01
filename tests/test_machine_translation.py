"""Test Machine Translation module"""

import unittest

from pororo import Pororo


class PororoTranslationTester(unittest.TestCase):

    def test_modules(self):
        # # Test Korean-English module
        # ko_en = Pororo(task="mt", lang="ko", tgt="en")
        # ko_en_res = ko_en("나는 여기에 산다.")
        # self.assertIsInstance(ko_en_res, str)

        # # Test English-Korean module
        # en_ko = Pororo(task="mt", lang="en", tgt="ko")
        # en_ko_res = en_ko("I live here.")
        # self.assertIsInstance(en_ko_res, str)

        # # Test Korean-Japanese module
        # ko_ja = Pororo(task="mt", lang="ko", tgt="ja")
        # ko_ja_res = ko_ja("나는 여기에 산다.")
        # self.assertIsInstance(ko_ja_res, str)

        # # Test Japanese-Korean module
        # ja_ko = Pororo(task="mt", lang="ja", tgt="ko")
        # ja_ko_res = ja_ko("私はここに住んでいる。")
        # self.assertIsInstance(ja_ko_res, str)

        # # Test Korean-Chinese module
        # ko_zh = Pororo(task="mt", lang="ko", tgt="zh")
        # ko_zh_res = ko_zh("나는 여기에 산다.")
        # self.assertIsInstance(ko_zh_res, str)

        # # Test Chinese-Korean module
        # zh_ko = Pororo(task="mt", lang="zh", tgt="ko")
        # zh_ko_res = zh_ko("我住在这里。")
        # self.assertIsInstance(zh_ko_res, str)

        # # Test Korean-Jejueo module
        # ko_je = Pororo(task="mt", lang="ko", tgt="je")
        # ko_je_res = ko_je("나는 여기에 산다.")
        # self.assertIsInstance(ko_je_res, str)

        # # Test Jejueo-Korean module
        # je_ko = Pororo(task="mt", lang="je", tgt="ko")
        # je_ko_res = je_ko("나는 여기에 산다뿌.")
        # self.assertIsInstance(je_ko_res, str)

        # # Test Chinese-Korean module
        # zh_en = Pororo(task="mt", lang="zh", tgt="en")
        # zh_en_res = zh_en("我住在这里。")
        # self.assertIsInstance(zh_en_res, str)

        # # Test Japanese-Korean module
        # ja_en = Pororo(task="mt", lang="ja", tgt="en")
        # ja_en_res = ja_en("私はここに住んでいる。")
        # self.assertIsInstance(ja_en_res, str)

        # Test Multi module
        multi = Pororo(task="mt", lang="multi")
        multi_res = multi("나는 여기에 산다.", src="ko", tgt="en")
        self.assertIsInstance(multi_res, str)


if __name__ == "__main__":
    unittest.main()
