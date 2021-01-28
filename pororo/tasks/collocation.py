"""Collocation related modeling class"""

from typing import Optional

from pororo.tasks.utils.base import PororoFactoryBase, PororoSimpleBase
from pororo.tasks.utils.download_utils import download_or_load


class PororoCollocationFactory(PororoFactoryBase):
    """
    Conduct collocation search using index file

    English (`collocate.en`)

        - dataset: enwiki-20180420
        - metric: N/A

    Korean (`kollocate`)

        - dataset: kowiki-20200720
        - metric: N/A

    Chinese (`collocate.zh`)

        - dataset: zhwiki-20180420
        - metric: N/A

    Japanse (`collocate.ja`)

        - dataset: jawiki-20180420
        - metric: N/A

    Args:
        text (str): text to be inputted for collocation search

    Returns:
        dict: searched collocation splitted by part of speech

    Examples:
        >>> col = Pororo(task="col", lang="ko")
        >>> col("먹")
        먹 as verb
        noun 것(39), 수(29), 음식(23), 등(16), 고기(14), ..
        verb 하(33), 않(21), 살(17), 즐기(11), 굽(9), ..
        adverb 많이(10), 주로(7), 다(5), 같이(4), 잘(4), ...
        determiner 다른(5), 그(2), 여러(1), 세(1), 몇몇(1), 새(1)
        adjective 싶(5), 어리(1), 편하(1), 작(1), 좋(1), 손쉽(1), 못하(1)

        먹 as noun
        noun 붓(3), 종이(2), 묘선(1), 청자(1), 은장도(1), 제조(1), ..
        verb 의하(1), 그리(1), 찍(1), 차(1), 늘어놓(1)
        adverb 하지만(1)

    """

    def __init__(self, task: str, lang: str, model: Optional[str]):
        super().__init__(task, lang, model)

    @staticmethod
    def get_available_langs():
        return ["ko", "en", "ja", "zh"]

    @staticmethod
    def get_available_models():
        return {
            "ko": ["kollocate"],
            "en": ["collocate"],
            "ja": ["collocate"],
            "zh": ["collocate"],
        }

    def load(self, device: str):
        """
        Load user-selected task-specific model

        Args:
            device (str): device information

        Returns:
            object: User-selected task-specific model

        """
        if self.config.n_model == "kollocate":
            try:
                from kollocate import Kollocate
            except ModuleNotFoundError as error:
                raise error.__class__(
                    "Please install kollocate with: `pip install kollocate`")
            model = Kollocate()
            return PororoCollocate(model, self.config)

        if self.config.n_model == "collocate":
            from pororo.models.collocate import Collocate

            index_path = download_or_load(
                f"misc/collocate.{self.config.lang}.zip",
                self.config.lang,
            )
            model = Collocate(index_path)
            return PororoCollocate(model, self.config)


class PororoCollocate(PororoSimpleBase):

    def __init__(self, model, config):
        super().__init__(config)
        self._model = model

    def predict(self, text: str) -> str:
        """
        Conduct collocation search using index file

        Args:
            text (str): text to be inputted for collocation search

        Returns:
            dict: searched collocation splitted by part of speech

        """
        return self._model(text)
