"""Grapheme to Phoneme related modeling class"""

import os
from typing import List, Optional, Union

from pororo.tasks.utils.base import PororoFactoryBase, PororoSimpleBase


class PororoG2pFactory(PororoFactoryBase):
    """
    Conduct grapheme to phoneme conversion

    English (`g2p.en`)

        - dataset: g2pE
        - metric: N/A
        - ref: https://github.com/Kyubyong/g2p

    Korean (`g2p.ko`)

        - dataset: g2pK
        - metric: N/A
        - ref: https://github.com/Kyubyong/g2pK

    Japanese (`g2p.ja`)

        - dataset: romkan
        - metric: N/A
        - ref: https://github.com/soimort/python-romkan

    Chinese (`g2p.zh`)

        - dataset: g2pM (Park et al. 2020)
        - metric: N/A
        - ref: https://github.com/kakaobrain/g2pM

    Args:
        sent (str): input sentence to be converted to phoneme
        align (bool): whether to align the result (not applied to `english` model)

    Returns:
        str: converted phoneme sentence

    Examples:
        >>> g2p = Pororo(task="g2p", lang="ko")
        >>> g2p("어제는 날씨가 맑았는데, 오늘은 흐리다.")
        '어제는 날씨가 말간는데, 오느른 흐리다.'
        >>> g2p("어제는 날씨가 맑았는데, 오늘은 흐리다.", align=True)
        [('어제는', '어제는'), ('날씨가', '날씨가'), ('맑았는데,', '말간는데,'), ('오늘은', '오느른'), ('흐리다.', '흐리다.')]
        >>> g2p_en = Pororo(task="g2p", lang="en")
        >>> g2p_en("I have $250 in my pocket.")
        ['AY1', ' ', 'HH', 'AE1', 'V', ' ', 'T', 'UW1', ' ', 'HH', 'AH1', 'N', 'D', 'R', 'AH0', 'D', ' ', 'F', 'IH1', 'F', 'T', 'IY0', ' ', 'D', 'AA1', 'L', 'ER0', 'Z', ' ', 'IH0', 'N', ' ', 'M', 'AY1', ' ', 'P', 'AA1', 'K', 'AH0', 'T', ' ', '.']
        >>> g2p_zh = Pororo(task="g2p", lang="zh")
        >>> g2p_zh("然而，他红了20年以后，他竟退出了大家的视线。")
        'ran2 er2 , ta1 hong2 le5 2 0 nian2 yi3 hou4 , ta1 jing4 tui4 chu1 le5 da4 jia1 de5 shi4 xian4 。'
        >>> g2p_zh("然而，他红了20年以后，他竟退出了大家的视线。", align=True)
        [('然', 'ran2'), ('而', 'er2'), (',', ','), ('他', 'ta1'), ('红', 'hong2'), ('了', 'le5'), ('2', '2'), ('0', '0'), ('年', 'nian2'), ...]
        >>> g2p_zh("然而，他红了20年以后，他竟退出了大家的视线。", align=True, tone=False)
        [('然', 'ran'), ('而', 'er'), (',', ','), ('他', 'ta'), ('红', 'hong'), ('了', 'le'), ('2', '2'), ('0', '0'), ('年', 'nian'), ...]
        >>> g2p_ja = Pororo(task="g2p", lang="ja")
        >>> g2p_ja("pythonが大好きです")
        'python ga daisuki desu'
        >>> g2p_ja("pythonが大好きです", align=True)
        [('python', 'python'), ('が', 'ga'), ('大好き', 'daisuki'), ('です', 'desu')]

    """

    def __init__(self, task: str, lang: str, model: Optional[str]):
        super().__init__(task, lang, model)

    @staticmethod
    def get_available_langs():
        return ["en", "ko", "zh", "ja"]

    @staticmethod
    def get_available_models():
        return {
            "en": ["g2p.en"],
            "ko": ["g2p.ko"],
            "zh": ["g2p.zh"],
            "ja": ["g2p.ja"],
        }

    def load(self, device: str):
        """
        Load user-selected task-specific model

        Args:
            device (str): device information

        Returns:
            object: User-selected task-specific model

        """
        if self.config.n_model == "g2p.ko":
            try:
                from g2pk import G2p as G2pK
            except ModuleNotFoundError as error:
                raise error.__class__(
                    "Please install g2pk with: `pip install g2pk`")
            model = G2pK()
            return PororoG2PKo(model, self.config)

        if self.config.n_model == "g2p.en":
            try:
                from g2p_en import G2p as G2pE
            except ModuleNotFoundError as error:
                raise error.__class__(
                    "Please install g2p_en with: `pip install g2p_en`")
            model = G2pE()
            return PororoG2PEn(model, self.config)

        if self.config.n_model == "g2p.zh":
            try:
                from g2pM import G2pM
            except ModuleNotFoundError as error:
                raise error.__class__(
                    "Please install g2pM with: `pip install g2pM`")
            model = G2pM()
            return PororoG2PZh(model, self.config)

        if self.config.n_model == "g2p.ja":
            try:
                import fugashi
            except ModuleNotFoundError as error:
                raise error.__class__(
                    "Please install fugashi with: `pip install fugashi`")

            try:
                import ipadic
            except ModuleNotFoundError as error:
                raise error.__class__(
                    "Please install ipadic with: `pip install ipadic`")

            try:
                import romkan
            except ModuleNotFoundError as error:
                raise error.__class__(
                    "Please install ipadic with: `pip install romkan`")
            dic_dir = ipadic.DICDIR
            mecabrc = os.path.join(dic_dir, "mecabrc")
            mecab_option = "-d {} -r {} ".format(
                dic_dir,
                mecabrc,
            )
            tagger = fugashi.GenericTagger(mecab_option)
            return PororoG2PJa(tagger, romkan.to_roma, self.config)


class PororoG2PKo(PororoSimpleBase):

    def __init__(self, model, config):
        super().__init__(config)
        self._model = model

    def predict(self, text: str, **kwargs) -> Union[List, str]:
        """
        Conduct grapheme to phoneme conversion

        Args:
            text (str): input sentence to be converted to phoneme
            align (bool): whether to align the result

        Returns:
            str: converted phoneme sentence

        """
        align = kwargs.get("align", False)

        results = self._model(text)

        if align:
            return [(word, phoneme)
                    for word, phoneme in zip(text.split(), results.split())]
        return results


class PororoG2PEn(PororoSimpleBase):

    def __init__(self, model, config):
        super().__init__(config)
        self._model = model

    def predict(self, text: str, **kwargs) -> str:
        """
        Conduct grapheme to phoneme conversion

        Args:
            sent (str): input sentence to be converted to phoneme

        Returns:
            str: converted phoneme sentence

        """
        return self._model(text)


class PororoG2PZh(PororoSimpleBase):

    def __init__(self, model, config):
        super().__init__(config)
        self._model = model

    def predict(self, text: str, **kwargs) -> Union[List, str]:
        """
        Conduct grapheme to phoneme conversion

        Args:
            text (str): input sentence to be converted to phoneme
            align (bool): whether to align the result
            tone (bool): whether to show chinese tone

        Returns:
            str: converted phoneme sentence

        """
        align = kwargs.get("align", False)
        tone = kwargs.get("tone", True)
        results = self._model(text, tone=tone, char_split=True)

        if align:
            return [(c, t) for c, t in zip(text, results)]
        return " ".join(results)


class PororoG2PJa(PororoSimpleBase):

    def __init__(self, tagger, romanize, config):
        super().__init__(config)
        self._tagger = tagger
        self._romanize = romanize

    def predict(self, text: str, **kwargs):
        """
        Conduct grapheme to phoneme conversion

        Args:
            text (str): input sentence to be converted to phoneme
            align (bool): whether to align the result

        Returns:
            List[str]: converted phoneme sentence list

        """
        align = kwargs.get("align", False)

        output = self._tagger.parse(text.strip())
        results = list()

        for line in output.splitlines():
            res = line.split("\t")

            if len(res) > 1:
                orig = res[0]
                feats = res[1].split(",")
                reading = feats[-1]

                if reading == "*":
                    reading = orig

                romaji = self._romanize(reading)
                results.append((orig, romaji))

        if align:
            return results

        return " ".join([result[1] for result in results])
