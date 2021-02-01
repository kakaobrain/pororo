"""Test ZSL module"""

import unittest

from pororo import Pororo


class PororoZSLTester(unittest.TestCase):

    def test_modules(self):
        zsl = Pororo(task="zero-topic", lang="ko")
        zsl_res = zsl(
            """라리가 사무국, 메시 아닌 바르사 지지..."바이 아웃 유효" [공식발표]""",
            [
                "스포츠",
                "사회",
                "정치",
                "경제",
                "생활/문화",
                "IT/과학",
            ],
        )
        self.assertIsInstance(zsl_res, dict)

        zsl = Pororo(task="zero-topic")
        zsl(
            "Who are you voting for in 2020?",
            [
                "business",
                "art & culture",
                "politics",
            ],
        )

        zsl = Pororo(task="zero-topic", lang="ja")
        zsl(
            "香川 真司は、兵庫県神戸市垂水区出身のプロサッカー選手。元日本代表。ポジションはMF、FW。ボルシア・ドルトムント時代の2010-11シーズンでリーグ前半期17試合で8得点を記録し9シーズンぶりのリーグ優勝に貢献。キッカー誌が選定したブンデスリーガの年間ベスト イレブンに名を連ねた。",
            [
                "スポーツ",
                "政治",
                "技術",
            ],
        )

        zsl = Pororo(task="zero-topic", lang="zh")
        zsl(
            "商务部14日发布数据显示，今年前10个月，我国累计对外投资904.6亿美元，同比增长5.9%。",
            [
                "政治",
                "经济",
                "国际化",
            ],
        )


if __name__ == "__main__":
    unittest.main()
