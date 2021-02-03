"""Review Scoring related modeling class"""

from typing import Optional

from pororo.tasks.utils.base import PororoFactoryBase, PororoSimpleBase


class PororoReviewFactory(PororoFactoryBase):
    """
    Regression based Review scoring using Review Corpus

    English (`roberta.base.en.review`)

        - dataset: Multilingual Amazon Reviews Corpus (Phillip Keung et al, 2019)
        - metric: Pearson (86.85), Spearman (86.60)

    Japanese (`jaberta.base.ja.review`)

        - dataset: Multilingual Amazon Reviews Corpus (Phillip Keung et al, 2019)
        - metric: Pearson (85.07), Spearman (85.05)

    Chinese (`zhberta.base.zh.review`)

        - dataset: Multilingual Amazon Reviews Corpus (Phillip Keung et al, 2019)
        - metric: Pearson (80.12), Spearman (80.01)

    Korean (`brainbert.base.ko.review_rating`)

        - dataset: Internal data
        - metric: Pearson (78.03), Spearman (77.93)

    Examples:
        >>> review = Pororo(task="review", lang="en")
        >>> review("Just what I needed! Perfect for western theme party.")
        4.79
        >>> review("Received wrong size.")
        2.65
        >>> review = Pororo(task="review", lang="ja")
        >>> review("充電あまりしません! 星5だったのに騙されました!")
        0.86
        >>> review("迅速な対応ありがとうございます。 今後ともよろしくお願いします。")
        4.7
        >>> review = Pororo(task="review", lang="zh")
        >>> review("买的两百多的，不是真货，和真的对比了小一圈！特别不好跟30多元的没区别，退货了！不建议买！")
        1.47
        >>> review("锅外型好可爱，家人喜欢，很适合3口之家使用")
        4.88
        >>> review = Pororo(task="review", lang="ko")
        >>> review("그냥저냥 다른데랑 똑같숩니다")
        2.96
        >>> review("좋습니다 만족해요 배송만 좀 더 빨랐으면..")
        3.92

    """

    def __init__(self, task: str, lang: str, model: Optional[str]):
        super().__init__(task, lang, model)

    @staticmethod
    def get_available_langs():
        return ["en", "ko", "ja", "zh"]

    @staticmethod
    def get_available_models():
        return {
            "en": ["roberta.base.en.review"],
            "ko": ["brainbert.base.ko.review_rating"],
            "ja": ["jaberta.base.ja.review"],
            "zh": ["zhberta.base.zh.review"],
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

            return PororoBertReviewScore(model, self.config)

        if "roberta" in self.config.n_model:
            from pororo.models.brainbert import CustomRobertaModel

            model = (CustomRobertaModel.load_model(
                f"bert/{self.config.n_model}",
                self.config.lang,
            ).eval().to(device))

            return PororoBertReviewScore(model, self.config)

        if "jaberta" in self.config.n_model:
            from pororo.models.brainbert import JabertaModel

            model = (JabertaModel.load_model(
                f"bert/{self.config.n_model}",
                self.config.lang,
            ).eval().to(device))

            return PororoBertReviewScore(model, self.config)

        if "zhberta" in self.config.n_model:
            from pororo.models.brainbert import ZhbertaModel

            model = (ZhbertaModel.load_model(
                f"bert/{self.config.n_model}",
                self.config.lang,
            ).eval().to(device))

            return PororoBertReviewScore(model, self.config)


class PororoBertReviewScore(PororoSimpleBase):

    def __init__(self, model, config):
        super().__init__(config)
        self._model = model

    def predict(self, sent: str, **kwargs) -> float:
        """
        Conduct review rating scaled from 1.0 to 5.0

        Args:
            sent: (str) sentence to be rated

        Returns:
            float: rating score scaled from 1.0 to 5.0

        """
        score = self._model.predict_output(sent) * 5
        return round(score, 2)
