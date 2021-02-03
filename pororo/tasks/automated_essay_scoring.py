"""Automated Essay Scoring related modeling class"""

from typing import Optional

from pororo.tasks.utils.base import PororoFactoryBase, PororoSimpleBase


class PororoAesFactory(PororoFactoryBase):
    """
    Regression based Automated Essay Scoring

    English (`roberta.base.en.aes`)

        - dataset: The Hewlett Foundation: Automated Essay Scoring
        - metric: Spearman (80.25)
        - ref: https://www.kaggle.com/c/asap-aes/data

    Examples:
        >>> aes = Pororo(task="aes", lang="en")
        >>> aes("To me, leadership does not necessarily mean accumulating as many titles as possible...")
        23.56

    """

    def __init__(self, task: str, lang: str, model: Optional[str]):
        super().__init__(task, lang, model)

    @staticmethod
    def get_available_langs():
        return ["en"]

    @staticmethod
    def get_available_models():
        return {
            "en": ["roberta.base.en.aes"],
        }

    def load(self, device):
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

            return PororoBertAes(model, self.config)


class PororoBertAes(PororoSimpleBase):

    def __init__(self, model, config):
        super().__init__(config)
        self._model = model

    def predict(self, sent: str, **kwargs):
        """
        Conduct Automated Essay Scoring

        Args:
            sent: (str) sentence to be encoded

        Returns:
            float: predicted essay score

        """
        tokens = self._model.encode(sent)
        score = (self._model.predict(
            "sentence_classification_head",
            tokens[:1024],
            return_logits=True,
        ).squeeze(-1).tolist()[0])
        return round(score * 100, 2)
