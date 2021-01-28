"""Test Automatic Speech Recognition module"""

import unittest

from pororo import Pororo
from pororo.utils import control_temp


class PororoASRTester(unittest.TestCase):

    # TODO: + MP3 file Test
    def test_modules(self):
        # yapf: disable
        asr = Pororo(task="asr", lang="ko")
        with control_temp("https://twg.kakaocdn.net/pororo/ko/example/korean_speech.wav") as f_src:
            asr_res = asr(f_src)
            self.assertIsInstance(asr_res, dict)

        asr_res = asr("https://www.youtube.com/watch?v=wIMttaKrtN0")
        self.assertIsInstance(asr_res, dict)

        asr = Pororo(task="asr", lang="en")
        with control_temp("https://twg.kakaocdn.net/pororo/en/example/english_speech.flac") as f_src:
            asr_res = asr(f_src)
            self.assertIsInstance(asr_res, dict)

        asr = Pororo(task="asr", lang="zh")
        with control_temp("https://twg.kakaocdn.net/pororo/zh/example/chinese_speech.wav") as f_src:
            asr_res = asr(f_src)
            self.assertIsInstance(asr_res, dict)
        # yapf: enable


if __name__ == "__main__":
    unittest.main()
