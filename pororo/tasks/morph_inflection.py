"""Morphological Inflection related modeling class"""

import pickle
from typing import Optional

from pororo.tasks.utils.base import PororoFactoryBase, PororoSimpleBase
from pororo.tasks.utils.download_utils import download_or_load


class PororoInflectionFactory(PororoFactoryBase):
    """
    Conduct Morphological inflection

    English (`enparadigm`)

        - dataset: TBU
        - metric: N/A

    Korean (`koparadigm`)

        - dataset: KoParadigm (Park et al. 2020)
        - metric: N/A

    Japanese (`japaradigm`)

        - dataset: TBU
        - metric: N/A

    Args:
        text (str): input text to be morphologically inflected

    Returns:
        List[Tuple[str, Tuple[str, str]]]: morphogical inflection token list

    Examples:
        >>> inflection = Pororo(task="inflection", lang="ko")
        >>> inflection("곱")
        [['Action Verb', [('거나', '곱거나'), ('거늘', '곱거늘'), ('거니', '곱거니') ...]]]
        >>> inflection = Pororo(task="inflection", lang="en")
        >>> inflection("love")
        {'NN': [('loves', 'NNS')], 'VB': [('loves', 'VBZ'), ('loved', 'VBD'), ('loved', 'VBN'), ('loving', 'VBG')]}
        >>> inflection = Pororo(task="inflection", lang="ja")
        >>> inflection("あえぐ")
        {'verb': [('あえが', '未然形'), ('あえご', '未然ウ接続'), ('あえぎ', '連用形'), ('あえい', '連用タ接続'), ('あえげ', '仮定形'), ('あえげ', '命令ｅ'), ('あえぎゃ', '仮定縮約１')]}

    """

    def __init__(self, task: str, lang: str, model: Optional[str]):
        super().__init__(task, lang, model)

    @staticmethod
    def get_available_langs():
        return ["en", "ko", "ja"]

    @staticmethod
    def get_available_models():
        return {
            "en": ["enparadigm"],
            "ko": ["koparadigm"],
            "ja": ["japaradigm"],
        }

    def load(self, device: int):
        """
        Load user-selected task-specific model

        Args:
            device (str): device information

        Returns:
            object: User-selected task-specific model

        """
        if self.config.n_model == "koparadigm":
            try:
                from koparadigm import Paradigm
            except ModuleNotFoundError as error:
                raise error.__class__(
                    "Please install koparadigm with: `pip install koparadigm`")
            model = Paradigm()
            return PororoKoParadigm(model, self.config)

        if self.config.n_model in ["enparadigm", "japaradigm"]:
            model_path = download_or_load(
                f"misc/inflection.{self.config.lang}.pickle",
                self.config.lang,
            )
            with open(model_path, "rb") as handle:
                model = dict(pickle.load(handle))
            return PororoParadigm(model, self.config)


class PororoKoParadigm(PororoSimpleBase):

    def __init__(self, model, config):
        super().__init__(config)
        self._model = model

    def predict(self, text: str):
        """
        Conduct korean morphological inflection

        Args:
            text (str): input text to be morphologically inflected

        Returns:
            List[Tuple[str, Tuple[str, str]]]: morphogical inflection token list

        """
        return self._model.conjugate(text)


class PororoParadigm(PororoSimpleBase):

    def __init__(self, model, config):
        super().__init__(config)
        self._model = model

    def predict(self, text: str, **kwargs):
        """
        Conduct morphological inflection

        Args:
            text (str): input text to be morphologically inflected

        Returns:
            List[Tuple[str, Tuple[str, str]]]: morphogical inflection token list

        """
        try:
            return self._model[text]
        except KeyError:
            raise KeyError("Un-registered key !")
