"""Test Dependency Parsing module"""

import unittest

from pororo import Pororo


class PororoDPTester(unittest.TestCase):

    def test_modules(self):
        dp = Pororo(task="dep_parse", lang="ko")
        dp_res = dp("[UCL 리뷰] '디마리아 1골 2도움' PSG, 라이프치히 3-0 제압...사상 첫 결승행")
        self.assertIsInstance(dp_res, list)


if __name__ == "__main__":
    unittest.main()
