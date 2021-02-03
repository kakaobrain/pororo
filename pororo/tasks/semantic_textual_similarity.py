"""Semantic Textual Similarity related modeling class"""

from typing import Optional

from scipy import spatial

from pororo.tasks.utils.base import PororoBiencoderBase, PororoFactoryBase
from pororo.tasks.utils.download_utils import download_or_load


class PororoStsFactory(PororoFactoryBase):
    """
    Sentence similarity base semantic textual similarity using korsts, sts

    Korean (`brainbert.base.ko.korsts`)

        - dataset: KorSTS (Ham et al. 2020)
        - metric: Spearman (83.00)

    Korean (`brainsbert.base.ko.kornli.korsts`)

        - dataset: KorSTS (Ham et al. 2020)
        - metric: Spearman (83.46)

    English (`roberta.base.en.sts`)

        - dataset: STS-B (Daniel Cer et al. 2017)
        - metric: Spearman (91.2)

    Japanese (`jaberta.base.ja.sts`)

        - dataset: Translated `STS-B` (Daniel Cer et al. 2017)
        - metric: Spearman (82.80)

    Chinese (`zhberta.base.zh.sts`)

        - dataset: Translated `STS-B` (Daniel Cer et al. 2017)
        - metric: Spearman (83.65)

    Examples:
        >>> sts = Pororo(task="similarity", lang="ko")
        >>> sts("나는 동물을 좋아하는 사람이야", "강아지를 좋아하는 아버지")
        0.415
        >>> sts = Pororo(task="similarity", lang="ja")
        >>> sts("ベビーパンダがスライドを下ります。", "パンダがスライドを下って滑ります。") # ["아기 팬더가 슬라이드를 내려 갑니다.", "팬더가 슬라이드를 내려 미끄러집니다."]
        0.746
        >>> sts = Pororo(task="similarity", lang="zh")
        >>> sts('三名男子在街上做同样的舞蹈。', '街上有三个无衬衫的男人在跳舞。')  # ["세 남자가 거리에서 같은 춤을 춥니다.", "거리에서 춤추는 세 명의 벗은 남자가 있습니다."]
        0.669
        >>> sts = Pororo(task="similarity", lang="en")
        >>> sts("Two dogs and one cat sitting on couch.", "Two dogs and a cat resting on a couch.")
        0.921

    """

    def __init__(self, task: str, lang: str, model: Optional[str]):
        super().__init__(task, lang, model)

    @staticmethod
    def get_available_langs():
        return ["en", "ko", "ja", "zh"]

    @staticmethod
    def get_available_models():
        return {
            "en": ["roberta.base.en.sts"],
            "ko": [
                "brainbert.base.ko.korsts",
                "brainsbert.base.ko.kornli.korsts",
            ],
            "ja": ["jaberta.base.ja.sts", "jasbert.base.ja.nli.sts"],
            "zh": ["zhberta.base.zh.sts", "zhsbert.base.zh.nli.sts"],
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
            return PororoBertSts(model, self.config)

        if "jaberta" in self.config.n_model:
            from pororo.models.brainbert import JabertaModel

            model = (JabertaModel.load_model(
                f"bert/{self.config.n_model}",
                self.config.lang,
            ).eval().to(device))
            return PororoBertSts(model, self.config)

        if "zhberta" in self.config.n_model:
            from pororo.models.brainbert import ZhbertaModel

            model = (ZhbertaModel.load_model(
                f"bert/{self.config.n_model}",
                self.config.lang,
            ).eval().to(device))
            return PororoBertSts(model, self.config)

        if "sbert" in self.config.n_model:
            from sentence_transformers import SentenceTransformer

            path = download_or_load(
                f"sbert/{self.config.n_model}",
                self.config.lang,
            )
            model = SentenceTransformer(path).eval().to(device)
            return PororoSBertSts(model, self.config)

        if "roberta" in self.config.n_model:
            from pororo.models.brainbert import CustomRobertaModel

            model = (CustomRobertaModel.load_model(
                f"bert/{self.config.n_model}",
                self.config.lang,
            ).eval().to(device))
            return PororoBertSts(model, self.config)


class PororoBertSts(PororoBiencoderBase):

    def __init__(self, model, config):
        super().__init__(config)
        self._model = model

    def predict(self, sent_a: str, sent_b: str):
        """
        Conduct semantic textual similarity task with BERT

        Args:
            sent_a (str): first sentence to be encoded
            sent_b (str): second sentence to be encoded

        Returns:
            float: similarity score

        """
        sim = self._model.predict_output(sent_a, sent_b)
        return float("{:.3f}".format(sim))


class PororoSBertSts(PororoBiencoderBase):

    def __init__(self, model, config):
        super().__init__(config)
        self._model = model

    def predict(self, sent_a: str, sent_b: str, **kwargs) -> float:
        """
        Conduct semantic textual similariry task with S-BERT

        Args:
            sent_a (str): first sentence to be encoded
            sent_b (str): second sentence to be encoded

        Returns:
            float: similarity score

        """
        encoded = self._model.encode([sent_a, sent_b])
        vec_a, vec_b = encoded[0], encoded[-1]
        sim = 1 - spatial.distance.cosine(vec_a, vec_b)
        return float("{:.3f}".format(sim))
