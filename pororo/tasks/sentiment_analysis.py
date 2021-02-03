"""Sentiment Analysis related modeling class"""

from typing import Optional

from pororo.tasks.utils.base import PororoFactoryBase, PororoSimpleBase


class PororoSentimentFactory(PororoFactoryBase):
    """
    Classification based sentiment analysis using Review Corpus

    Korean (`brainbert.base.ko.shopping`)

        - dataset: Shopping review corpus
        - metric: Accuracy (95.00)
        - ref: https://github.com/bab2min/corpus/tree/master/sentiment

    Korean (`brainbert.base.ko.nsmc`)

        - dataset: Naver sentiment movie corpus
        - metric: Accuracy (90.84)
        - ref: https://github.com/e9t/nsmc

    Japanese (`jaberta.base.ja.sentiment`)

        - data: Internal data
        - metric: Accuracy (96.29)

    Examples:
        >>> sa = Pororo(task="sentiment", model="brainbert.base.ko.nsmc", lang="ko")
        >>> sa("배송이 버트 학습시키는 것 만큼 느리네요")
        'Negative'
        >>> sa("배송이 경량화되었는지 빠르네요")
        'Positive'
        >>> sa = Pororo(task="sentiment", lang="ja")
        >>> sa("日が暑くもイライラか。")  # 날이 더워서 너무 짜증나요.
        'Negative'
        >>> sa('日が良く散歩に行きたいです。')  # 날이 좋아서 산책을 가고 싶어요.
        'Positive'
        >>> sa = Pororo(task="sentiment", model="brainbert.base.ko.shopping", lang="ko")
        >>> sa("꽤 맘에 들었어요. 겉에서 봤을땐 허름?했는데 맛도 있고, 괜찮아요")
        'Positive'
        >>> sa("예약하고 가세요 대기줄이 깁니다 훠궈는 하이디라오가 비싼만큼 만족도가 제일 높아요")
        'Negative'
        >>> sa("이걸 산 내가 레전드", show_probs=True)
        {'negative': 0.7525266408920288, 'positive': 0.2474733293056488}

    """

    def __init__(self, task: str, lang: str, model: Optional[str]):
        super().__init__(task, lang, model)

    @staticmethod
    def get_available_langs():
        return ["ko", "ja"]

    @staticmethod
    def get_available_models():
        return {
            "ko": [
                "brainbert.base.ko.shopping",
                "brainbert.base.ko.nsmc",
            ],
            "ja": ["jaberta.base.ja.sentiment"],
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

            model = (BrainRobertaModel.load_model(
                f"bert/{self.config.n_model}",
                self.config.lang,
            ).eval().to(device))
            return PororoBertSentiment(model, self.config)

        if "jaberta" in self.config.n_model:
            from pororo.models.brainbert import JabertaModel

            model = (JabertaModel.load_model(
                f"bert/{self.config.n_model}",
                self.config.lang,
            ).eval().to(device))
            return PororoBertSentiment(model, self.config)


class PororoBertSentiment(PororoSimpleBase):

    def __init__(self, model, config):
        super().__init__(config)
        self._model = model
        self._label_fn = {
            "0": "negative",
            "1": "positive",
        }

    def predict(self, sent: str, **kwargs) -> str:
        """
        Conduct sentiment analysis

        Args:
            sent: (str) sentence to be sentiment analyzed
            show_probs: (bool) whether to show probability score

        Returns:
            str: predicted sentence label - `negative` or `positive`

        """
        show_probs = kwargs.get("show_probs", False)

        res = self._model.predict_output(sent, show_probs=show_probs)
        if show_probs:
            probs = {self._label_fn[r]: res[r] for r in res}
            return probs
        else:
            if self.config.lang == "ko":
                return self._label_fn[res].title()
            return res.title()
