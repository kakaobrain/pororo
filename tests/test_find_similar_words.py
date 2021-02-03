"""Test Similar Word Finding module"""

import unittest
from collections import OrderedDict

from pororo import Pororo


class PororoFindWordTester(unittest.TestCase):

    def test_modules(self):
        w2v = Pororo(task="wordvec", lang="ko")
        w2v_res = w2v.find_similar_words("사과")
        self.assertIsInstance(w2v_res, OrderedDict)

        w2v = Pororo("word2vec", lang="en")
        w2v_res = w2v.find_similar_words("apple", top_n=3, group=True)
        self.assertIsInstance(w2v_res, OrderedDict)

        w2v = Pororo("word2vec", lang="ja")
        w2v_res = w2v.find_similar_words("リンゴ")
        self.assertIsInstance(w2v_res, OrderedDict)

        w2v = Pororo("word2vec", lang="zh")
        w2v_res = w2v.find_similar_words("苹果")
        self.assertIsInstance(w2v_res, OrderedDict)


if __name__ == "__main__":
    unittest.main()
