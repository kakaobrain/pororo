"""Test heuristic library related pipeline modules"""

import unittest

from pororo.utils import postprocess_span


class PororoUtilFuncTester(unittest.TestCase):

    def test_modules(self):
        try:
            import mecab
            tagger = mecab.MeCab()

            res = postprocess_span(tagger, "이민자들은")
            self.assertIsInstance(res, str)
            self.assertEqual(res, "이민자들")

            res = postprocess_span(tagger, "8100억원에")
            self.assertIsInstance(res, str)
            self.assertEqual(res, "8100억원")

            res = postprocess_span(tagger, "1960년대부터")
            self.assertIsInstance(res, str)
            self.assertEqual(res, "1960년대")

            res = postprocess_span(tagger, "군사 목적으로는,")
            self.assertIsInstance(res, str)
            self.assertEqual(res, "군사 목적")

        except ModuleNotFoundError as error:
            raise error.__class__(
                "Please install python-mecab-ko with: `pip install python-mecab-ko`"
            )


if __name__ == "__main__":
    unittest.main()
