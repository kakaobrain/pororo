"""Test Tokenization module"""

import unittest

from pororo import Pororo


class PororoTokenizerTester(unittest.TestCase):

    def test_modules(self):
        mecab = Pororo(task="tokenize", lang="ko", model="mecab_ko")
        mecab_res = mecab("안녕 나는 민이라고 해.")
        self.assertIsInstance(mecab_res, list)

        bpe = Pororo(task="tokenise", lang="ko", model="bpe32k.ko")
        bpe_res = bpe("안녕 나는 민이라고 해.")
        self.assertIsInstance(bpe_res, list)

        unigram = Pororo(task="tokenization", lang="ko", model="unigram32k.ko")
        unigram_res = unigram("안녕 나는 민이라고 해.")
        self.assertIsInstance(unigram_res, list)

        char = Pororo(task="tokenization", lang="ko", model="char")
        char_res = char("안녕 나는 민이라고 해.")
        self.assertIsInstance(char_res, list)

        jamo = Pororo(task="tokenization", lang="ko", model="jamo")
        jamo_res = jamo("안녕 나는 민이라고 해.")
        self.assertIsInstance(jamo_res, list)

        jpe = Pororo(task="tokenization", lang="ko", model="jpe32k.ko")
        jpe_res = jpe("안녕 나는 민이라고 해.")
        self.assertIsInstance(jpe_res, list)

        mecab_bpe = Pororo(
            task="tokenization",
            lang="ko",
            model="mecab.bpe32k.ko",
        )
        mecab_bpe_res = mecab_bpe("안녕 나는 민이라고 해.")
        self.assertIsInstance(mecab_bpe_res, list)


if __name__ == "__main__":
    unittest.main()
