"""Test Speech Translation module"""

import unittest

from pororo import Pororo
from pororo.utils import control_temp


class PororoSTTester(unittest.TestCase):

    def test_modules(self):
        # yapf: disable
        st = Pororo(task="st", lang="ko")
        with control_temp("https://twg.kakaocdn.net/pororo/ko/example/korean_speech.wav") as f_src:
            st_res = st(f_src, tgt="en")
            self.assertIsInstance(st_res, dict)
        # yapf: enable


if __name__ == "__main__":
    unittest.main()
