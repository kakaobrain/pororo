"""Test Image Captioning module"""

import unittest

from pororo import Pororo


class PororoCaptionTester(unittest.TestCase):

    def test_modules(self):
        caption = Pororo(task="caption", lang="en")
        caption_res = caption(
            "https://i.pinimg.com/originals/b9/de/80/b9de803706fb2f7365e06e688b7cc470.jpg"
        )
        self.assertIsInstance(caption_res, str)


if __name__ == "__main__":
    unittest.main()
