"""Test Constituency Parsing module"""

import unittest

from pororo import Pororo


class PororoConstParsingTester(unittest.TestCase):

    def test_modules(self):
        const = Pororo(task="const", lang="ko")
        const_res = const("안녕하세요. 제 이름은 카터입니다.")
        self.assertIsInstance(const_res, str)

        const_res = const('이 감독은 "와인을 주신다는데 어떻게 빈 손으로 맞는가"라며 웃었다.')
        self.assertIsInstance(const_res, str)

        const_res = const(
            "지금까지 최원호 한화 이글스 감독대행, 이동욱 NC 다이노스 감독, 이강철 KT 감독에 이어 4번째 선물이었다.")
        self.assertIsInstance(const_res, str)


if __name__ == "__main__":
    unittest.main()
