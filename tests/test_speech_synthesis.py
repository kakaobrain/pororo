"""Test Text-To-Speech module"""

import unittest

import numpy as np

from pororo import Pororo


class PororoTTSTester(unittest.TestCase):

    def test_modules(self):
        tts = Pororo(task="tts", lang="multi")
        wave = tts("how are you?", lang="en", speaker="en")
        self.assertIsInstance(wave, np.ndarray)

        tts = Pororo(task="tts", lang="multi")
        wave = tts("저는 미국 사람이에요.", lang="ko", speaker="en")
        self.assertIsInstance(wave, np.ndarray)


if __name__ == "__main__":
    unittest.main()
