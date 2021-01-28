"""Test Reading Comprehension module"""

import unittest

from pororo import Pororo


class PororoMrcTester(unittest.TestCase):

    def test_modules(self):
        mrc = Pororo(task="mrc", lang="ko")
        mrc_res = mrc(
            "카카오브레인이 공개한 것은?",
            """카카오 인공지능(AI) 연구개발 자회사 카카오브레인이 AI 솔루션을 첫 상품화했다. 카카오는 카카오브레인 '포즈(pose·자세분석) API'를 유료 공개한다고 24일 밝혔다. 카카오브레인이 AI 기술을 유료 API를 공개하는 것은 처음이다. 공개하자마자 외부 문의가 쇄도한다. 포즈는 AI 비전(VISION, 영상·화면분석) 분야 중 하나다. 카카오브레인 포즈 API는 이미지나 영상을 분석해 사람 자세를 추출하는 기능을 제공한다.""",
        )
        self.assertIsInstance(mrc_res, tuple)


if __name__ == "__main__":
    unittest.main()
