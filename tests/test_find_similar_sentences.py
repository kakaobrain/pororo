"""Test Similar Sentence Finding module"""

import unittest

from pororo import Pororo


class PororoFindSentenceTester(unittest.TestCase):

    def test_modules(self):
        neighbor = Pororo(task="sentvec", lang="ko")
        neighbor_res = neighbor.find_similar_sentences(
            "프로그래머가 컴퓨터를 하고 있다.",
            [
                "고양이 두 마리가 서로 싸우고 있다.",
                "여성과 남성이 길을 걷고 있다.",
                "창문이 열려 있다.",
                "펼쳐져 있는 책 위에 연필이 있다.",
                "프로그래머가 코딩 중이다.",
                "참새와 제비가 날아다니고 있다.",
                "안경 쓴 여자가 아무 말을 적고 있다.",
                "컴퓨터를 하고 있는 사람",
                "아이들이 술래잡기를 하고 있다.",
            ],
        )
        self.assertIsInstance(neighbor_res, dict)


if __name__ == "__main__":
    unittest.main()
