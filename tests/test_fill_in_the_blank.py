"""Test Fill-in-the-blank module"""

import unittest

from pororo import Pororo


class PororoBlankTester(unittest.TestCase):

    def test_modules(self):
        fill = Pororo(task="fib", lang="en")
        fill_res = fill("David Beckham is a famous __ player.")
        self.assertIsInstance(fill_res, list)

        fill = Pororo(task="fib", lang="ko")
        fill_res = fill("아 그거 __으로 보내줘 ㅋㅋ")
        self.assertIsInstance(fill_res, list)

        fill = Pororo(task="fib", lang="zh")
        fill_res = fill("三__男子在街上做同样的舞蹈。")
        self.assertIsInstance(fill_res, list)

        fill = Pororo(task="fib", lang="ja")
        fill_res = fill("文在寅は__の大統領だ。")
        self.assertIsInstance(fill_res, list)


if __name__ == "__main__":
    unittest.main()
