"""Zero-shot Classification related modeling class"""

from typing import Dict, List, Optional

from pororo.tasks.utils.base import PororoBiencoderBase, PororoFactoryBase


class PororoZeroShotFactory(PororoFactoryBase):
    """
    Zero-shot topic classification

    See also:
        https://joeddav.github.io/blog/2020/05/29/ZSL.html

    Korean (`brainbert.base.ko.kornli`)

        - dataset: KorNLI (Ham et al. 2020)
        - metric: N/A

    English (`roberta.base.en.nli`)

        - dataset: MNLI (Adina Williams et al. 2017)
        - metric: N/A

    Japanese (`jaberta.base.ja.nli`)

        - dataset: XNLI (Alexis Conneau et al. 2018)
        - metric: N/A

    Chinese (`zhberta.base.zh.nli`)

        - dataset: XNLI (Alexis Conneau et al. 2018)
        - metric: N/A

    Examples:
        >>> zsl = Pororo(task="zero-topic")
        >>> zsl("Who are you voting for in 2020?", ["business", "art & culture", "politics"])
        {'business': 33.23, 'art & culture': 8.33, 'politics': 96.12}
        >>> zsl = Pororo(task="zero-topic", lang="ko")
        >>> zsl('''라리가 사무국, 메시 아닌 바르사 지지..."바이 아웃 유효" [공식발표]''', ["스포츠", "사회", "정치", "경제", "생활/문화", "IT/과학"])
        {'스포츠': 94.15, '사회': 37.11, '정치': 74.26, '경제': 39.18, '생활/문화': 71.15, 'IT/과학': 34.71}
        >>> zsl('''장제원, 김종인 당무감사 추진에 “참 잔인들 하다”···정강정책 개정안은 “졸작”''', ["스포츠", "사회", "정치", "경제", "생활/문화", "IT/과학"])
        {'스포츠': 2.18, '사회': 56.1, '정치': 88.24, '경제': 16.17, '생활/문화': 66.13, 'IT/과학': 11.2}
        >>> zsl = Pororo(task="zero-topic", lang="ja")
        >>> zsl("香川 真司は、兵庫県神戸市垂水区出身のプロサッカー選手。元日本代表。ポジションはMF、FW。ボルシア・ドルトムント時代の2010-11シーズンでリーグ前半期17試合で8得点を記録し9シーズンぶりのリーグ優勝に貢献。キッカー誌が選定したブンデスリーガの年間ベスト イレブンに名を連ねた。", ["スポーツ", "政治", "技術"])
        {'スポーツ': 0.2, '政治': 99.71, '技術': 68.9}
        >>> zsl = Pororo(task="zero-topic", lang="zh")
        >>> zsl("商务部14日发布数据显示，今年前10个月，我国累计对外投资904.6亿美元，同比增长5.9%。", ["政治", "经济", "国际化"])
        {'政治': 33.72, '经济': 3.9, '国际化': 13.67}

    """

    def __init__(self, task: str, lang: str, model: Optional[str]):
        super().__init__(task, lang, model)

    @staticmethod
    def get_available_langs():
        return ["en", "ko", "ja", "zh"]

    @staticmethod
    def get_available_models():
        return {
            "ko": ["brainbert.base.ko.kornli"],
            "ja": ["jaberta.base.ja.nli"],
            "zh": ["zhberta.base.zh.nli"],
            "en": ["roberta.base.en.nli"],
        }

    def load(self, device: str):
        """
        Load user-selected task-specific model

        Args:
            device (str): device information

        Returns:
            object: User-selected task-specific model

        """
        if "brainbert" in self.config.n_model:
            from pororo.models.brainbert import BrainRobertaModel

            model = BrainRobertaModel.load_model(
                f"bert/{self.config.n_model}",
                self.config.lang,
            ).eval().to(device)
            return PororoBertZeroShot(model, self.config)

        if "jaberta" in self.config.n_model:
            from pororo.models.brainbert import JabertaModel

            model = JabertaModel.load_model(
                f"bert/{self.config.n_model}",
                self.config.lang,
            ).eval().to(device)
            return PororoBertZeroShot(model, self.config)

        if "zhberta" in self.config.n_model:
            from pororo.models.brainbert import ZhbertaModel

            model = ZhbertaModel.load_model(
                f"bert/{self.config.n_model}",
                self.config.lang,
            ).eval().to(device)
            return PororoBertZeroShot(model, self.config)

        if "roberta" in self.config.n_model:
            from pororo.models.brainbert import CustomRobertaModel

            model = CustomRobertaModel.load_model(
                f"bert/{self.config.n_model}",
                self.config.lang,
            ).eval().to(device)
            return PororoBertZeroShot(model, self.config)


class PororoBertZeroShot(PororoBiencoderBase):

    def __init__(self, model, config):
        super().__init__(config)
        self._model = model
        self._template = {
            "ko": "이 문장은 {label}에 관한 것이다.",
            "ja": "この文は、{label}に関するものである。",
            "zh": "这句话是关于{label}的。",
            "en": "This sentence is about {label}.",
        }

    def predict(
        self,
        sent: str,
        labels: List[str],
        **kwargs,
    ) -> Dict[str, float]:
        """
        Conduct zero-shot classification

        Args:
            sent (str): sentence to be classified
            labels (List[str]): candidate labels

        Returns:
            List[Tuple(str, float)]: confidence scores corresponding to each input label

        """
        cands = [
            self._template[self.config.lang].format(label=label)
            for label in labels
        ]

        result = dict()
        for label, cand in zip(labels, cands):
            if self.config.lang == "ko":
                tokens = self._model.encode(
                    sent,
                    cand,
                    add_special_tokens=True,
                    no_separator=False,
                )
            else:
                tokens = self._model.encode(
                    sent,
                    cand,
                    no_separator=False,
                )

            # throw away "neutral" (dim 1) and take the probability of "entail" (2) as the probability of the label being true
            pred = self._model.predict(
                "sentence_classification_head",
                tokens,
                return_logits=True,
            )[:, [0, 2]]
            prob = pred.softmax(dim=1)[:, 1].item() * 100
            result[label] = round(prob, 2)

        return result
