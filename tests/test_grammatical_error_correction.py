"""Test Grammatical Error Correction module"""

import unittest

from pororo import Pororo


class PororoCounselingTester(unittest.TestCase):

    def test_modules(self):
        gec_en = Pororo(task="gec", lang="en")
        gec_res = gec_en("Myna me are kevi n")
        self.assertIsInstance(gec_res, str)

        gec_ko = Pororo(task="gec", lang="ko")
        gec_res = gec_ko("이걸이 렇게 한다 고?")
        self.assertIsInstance(gec_res, str)


if __name__ == "__main__":
    unittest.main()
