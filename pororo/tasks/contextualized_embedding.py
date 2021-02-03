"""Contextualized Embedding related modeling class"""

from typing import Optional

from pororo.tasks.utils.base import PororoFactoryBase, PororoSimpleBase


class PororoContextualFactory(PororoFactoryBase):
    """
    Conduct contextualized embedding

    English (`roberta.base.en`)

        - dataset: N/A
        - metric: N/A

    Korean (`brainbert.base.ko`)

        - dataset: N/A
        - metric: N/A

    Japanese (`jaberta.base.ja`)

        - dataset: N/A
        - metric: N/A

    Chinese (`zhberta.base.zh`)

        - dataset: N/A
        - metric: N/A

    Args:
        sent (str): input sentence to be contextualized embedded

    Returns:
        np.array: sentence embedding with subword units

    Examples:
        >>> cse = Pororo(task="cse", lang="ko")
        >>> cse("하늘을 나는 새")
        array([[92.53, 20.24, 32.32, ...],
            ...,
            [63.24, 53.19, 45.78, ...]], dtype=float32)  # (len(subwords), hidden_dim)
        >>> cse = Pororo(task="cse", lang="zh")
        >>> cse("一群人抬头看着建筑物屋顶边缘的3人。")
        array([[ 0.61136365,  0.24613665,  0.6259908 , ...,  0.32798234,
                0.10512973, -0.06808531],...,
            [-0.00931012, -0.04459633,  1.0253953 , ...,  0.30732906,
            0.22213839,  0.25226325]], dtype=float32)
        >>> cse = Pororo(task="cse", lang="ja")
        >>> cse("おはようございます")
        array([[-0.26724914, -0.23364174, -0.07206455, ...,  0.30293447,
                -0.36008322,  0.24684878], ...,
            [-0.7470922 , -0.30342472, -0.64015895, ..., -0.17556943,
                0.10660946, -0.17191087]], dtype=float32)

    """

    def __init__(self, task: str, lang: str, model: Optional[str]):
        super().__init__(task, lang, model)

    @staticmethod
    def get_available_langs():
        return ["en", "ko", "zh", "ja"]

    @staticmethod
    def get_available_models():
        return {
            "en": ["roberta.base.en"],
            "ko": ["brainbert.base.ko"],
            "zh": ["zhberta.base.zh"],
            "ja": ["jaberta.base.ja"],
        }

    def load(self, device: str):
        """
        Load user-selected task-specific model

        Args:
            device (str): device information

        Returns:
            object: User-selected task-specific model

        """
        if "roberta" in self.config.n_model:
            from pororo.models.brainbert import CustomRobertaModel

            model = (CustomRobertaModel.load_model(
                f"bert/{self.config.n_model}",
                self.config.lang,
            ).eval().to(device))
            return PororoBertContextualized(model, self.config, device)

        if "brainbert" in self.config.n_model:
            from pororo.models.brainbert import BrainRobertaModel

            model = BrainRobertaModel.load_model(
                f"bert/{self.config.n_model}",
                self.config.lang,
            ).eval().to(device)
            return PororoBertContextualized(model, self.config, device)

        if "jaberta" in self.config.n_model:
            from pororo.models.brainbert import JabertaModel

            model = JabertaModel.load_model(
                f"bert/{self.config.n_model}",
                self.config.lang,
            ).eval().to(device)
            return PororoBertContextualized(model, self.config, device)

        if "zhberta" in self.config.n_model:
            from pororo.models.brainbert import ZhbertaModel

            model = ZhbertaModel.load_model(
                f"bert/{self.config.n_model}",
                self.config.lang,
            ).eval().to(device)
            return PororoBertContextualized(model, self.config, device)


class PororoBertContextualized(PororoSimpleBase):

    def __init__(self, model, config, device):
        super().__init__(config)
        self._model = model
        self._device = device

    def predict(self, sent: str, **kwargs):
        """
        Conduct contextualized embedding

        Args:
            sent (str): input sentence to be contextualized embedded

        Returns:
            np.array: sentence embedding with subword units

        """
        indices = self._model.encode(sent).to(self._device)
        features, _ = self._model.model(
            indices.unsqueeze(0),
            features_only=True,
        )
        return features.squeeze(0).detach().cpu().numpy()
