"""Test Age Suitability module"""

import unittest

from pororo import Pororo


class PororoAgeSuitabilityTester(unittest.TestCase):

    def test_modules(self):
        age_suitability = Pororo(task="age_suitability", lang="en")
        age_res = age_suitability(
            "To me, leadership does not necessarily mean accumulating as many titles as possible..."
        )
        self.assertIsInstance(age_res, dict)


if __name__ == "__main__":
    unittest.main()
