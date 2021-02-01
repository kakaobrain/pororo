"""Test Pororo module"""

import unittest

from pororo import Pororo


class PororoPororoTester(unittest.TestCase):

    def test_modules(self):
        avail_tasks = Pororo.available_tasks()
        self.assertIsInstance(avail_tasks, str)

        avail_models = Pororo.available_models("ner")
        self.assertIsInstance(avail_models, str)


if __name__ == "__main__":
    unittest.main()
