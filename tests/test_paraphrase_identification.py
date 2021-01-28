"""Test Paraphrase Identification module"""

import unittest

from pororo import Pororo


class PororoParaIdTester(unittest.TestCase):

    def test_modules(self):
        para = Pororo(task="para", lang="ko")
        para_res = para("나는 천재다", "나는 바보다")
        self.assertIsInstance(para_res, str)


if __name__ == "__main__":
    unittest.main()
