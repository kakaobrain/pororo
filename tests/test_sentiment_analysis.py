"""Test Sentiment Analysis module"""

import unittest

from pororo import Pororo


class PororoSentimentTester(unittest.TestCase):

    def test_modules(self):
        sentiment = Pororo(task="sentiment", lang="ko")
        sentiment_res = sentiment("진짜 재미있다")
        self.assertIsInstance(sentiment_res, str)


if __name__ == "__main__":
    unittest.main()
