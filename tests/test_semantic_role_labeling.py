"""Test Semantic Role Labeling module"""

import unittest

from pororo import Pororo


class PororoSRLTester(unittest.TestCase):

    def test_modules(self):
        srl = Pororo(task="srl", lang="ko")
        srl_res = srl("카터는 역삼에서 카카오브레인으로 출근한다.")
        self.assertIsInstance(srl_res, list)


if __name__ == "__main__":
    unittest.main()
