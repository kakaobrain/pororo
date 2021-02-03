"""Natural Language Inference related modeling class"""

from typing import Optional

from pororo.tasks.utils.base import PororoBiencoderBase, PororoFactoryBase


class PororoNliFactory(PororoFactoryBase):
    """
    Classification based Natural Language Inference using KorNLI, MNLI, SNLI dataset

    English (`roberta.base.en.nli`)

        - dataset: MNLI (Adina Williams et al. 2017)
        - metric: Accuracy (87.6)

    Korean (`brainbert.base.ko.kornli`)

        - dataset: KorNLI (Ham et al. 2020)
        - metric: Accuracy (82.75)

    Japanese (`jaberta.base.ja.nli`)

        - dataset: XNLI (Alexis Conneau et al. 2018)
        - metric: Accuracy (85.27)

    Chinese (`zhberta.base.zh.nli`)

        - dataset: XNLI (Alexis Conneau et al. 2018)
        - metric: Accuracy (84.25)

    Examples:
        >>> nli = Pororo(task="nli", lang="ko")
        >>> nli("저는, 그냥 알아내려고 거기 있었어요", "나는 처음부터 그것을 잘 이해했다")
        'Contradiction'
        >>> nli = Pororo(task="nli", lang="ja")
        # ["오래된 신사는 세탁을하면서 사진을 찍고있는 것이 유머러스하다는 것을 알 수 있습니다.", "세탁을하면서 남자가 웃는"]
        >>> nli('古い紳士は、洗濯をしながら写真を撮っていることがユーモラスであることがわかります。', '洗濯をしながら男が笑う')
        'Entailment'
        # ["한 무리의 사람들이 건물 지붕 가장자리에있는 세 사람을 올려다 보았습니다.", "세 사람이 계단을 내려 가고 있습니다."]
        >>> nli = Pororo(task="nli", lang="zh")
        >>> nli('一群人抬头看着建筑物屋顶边缘的3人。', '三人正在楼梯上爬下来。')
        'Contradiction'
        >>> nli = Pororo(task="nli", lang="en")
        >>> nli("A soccer game with multiple males playing.", "Some men are playing a sport.")
        'Entailment'
    """

    def __init__(self, task: str, lang: str, model: Optional[str]):
        super().__init__(task, lang, model)

    @staticmethod
    def get_available_langs():
        return ["en", "ko", "ja", "zh"]

    @staticmethod
    def get_available_models():
        return {
            "en": ["roberta.base.en.nli"],
            "ko": ["brainbert.base.ko.kornli"],
            "ja": ["jaberta.base.ja.nli"],
            "zh": ["zhberta.base.zh.nli"],
        }

    def load(self, device):
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
            return PororoBertNli(model, self.config)

        if "jaberta" in self.config.n_model:
            from pororo.models.brainbert import JabertaModel

            model = (JabertaModel.load_model(
                f"bert/{self.config.n_model}",
                self.config.lang,
            ).eval().to(device))
            return PororoBertNli(model, self.config)

        if "zhberta" in self.config.n_model:
            from pororo.models.brainbert import ZhbertaModel

            model = (ZhbertaModel.load_model(
                f"bert/{self.config.n_model}",
                self.config.lang,
            ).eval().to(device))
            return PororoBertNli(model, self.config)

        if "roberta" in self.config.n_model:
            from pororo.models.brainbert import CustomRobertaModel

            model = (CustomRobertaModel.load_model(
                f"bert/{self.config.n_model}",
                self.config.lang,
            ).eval().to(device))
            return PororoBertNli(model, self.config)


class PororoBertNli(PororoBiencoderBase):

    def __init__(self, model, config):
        super().__init__(config)
        self._model = model

    def predict(
        self,
        sent_a: str,
        sent_b: str,
        **kwargs,
    ):
        """
        Conduct Natural Language Inference

        Args:
            sent_a: (str) first sentence to be encoded
            sent_b: (str) second sentence to be encoded

        Returns:
            str: predicted NLI label - Neutral, Entailment, or Contradiction

        """
        return self._model.predict_output(sent_a, sent_b).capitalize()
