"""Test Automated Essay Scoring module"""

import unittest

from pororo import Pororo


class PororoAESTester(unittest.TestCase):

    def test_modules(self):
        aes = Pororo(task="aes", lang="en")
        aes_res = aes(
            "To me, leadership does not necessarily mean accumulating as many titles as possible..."
        )
        self.assertIsInstance(aes_res, float)


if __name__ == "__main__":
    unittest.main()
