"""Word Translation related modeling class"""

from collections import namedtuple
from typing import Optional

from pororo.tasks.utils.base import PororoFactoryBase, PororoSimpleBase


class PororoWordTranslationFactory(PororoFactoryBase):
    """
    Conduct cross-word translation tasks

    See also:
        word2word: A Collection of Bilingual Lexicons for 3,564 Language Pairs (https://arxiv.org/abs/1911.12019)

    Multi (`word2word`)

        - dataset: word2word (Choe et al. 2019)
        - metric: N/A

    Args:
        word (str): input word to be translated

    Returns:
        List[str]: word translation candidates

    Examples:
        >>> wt = Pororo(task="word_translation", lang="en", tgt="fr")
        >>> wt("apple")
        ['pomme', 'pommier', 'pommes', 'tombe', 'yeux']
        >>> wt = Pororo(task="word_translation", lang="ja", tgt="ko")
        >>> wt("リンゴ")
        ['선악과', '사과', '에덴', '링고', '귀하']
        >>> wt = Pororo(task="word_translation", lang="ko", tgt="en")
        >>> wt("사과")
        ['apologize', 'apology', 'apple', 'apologies', 'apologizing']

    """

    # TODO: How to follow other module's __init__ concept?
    def __init__(self, task: str, lang: str, model: Optional[str], tgt: str):
        if model is None:
            model = "word2word"

        assert model in ["word2word"]

        Config = namedtuple("Config", ["task", "lang", "n_model"])
        self.config = Config(task, lang, model)

        self._tgt = tgt

    def load(self, device: str):
        """
        Load user-selected task-specific model

        Args:
            device (str): device information

        Returns:
            object: User-selected task-specific model

        """
        from word2word import Word2word

        model = Word2word(self.config.lang, self._tgt)
        return PororoWord2Word(model, self.config)


class PororoWord2Word(PororoSimpleBase):

    def __init__(self, model, config):
        super().__init__(config)
        self._model = model

    def predict(self, word: str, **kwargs):
        """
        Conduct word-to-word translation

        Args:
            word (str): input word to be translated

        Returns:
            List[str]: word translation candidates

        """
        return self._model(word)
