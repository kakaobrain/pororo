"""Part-Of-Speech Tagging related modeling class"""

import os
import re
from typing import List, Optional, Tuple, Union

from pororo.tasks.utils.base import PororoFactoryBase, PororoSimpleBase


class PororoPosFactory(PororoFactoryBase):
    """
    Conduct Part-of-Speech tagging

    Korean (`mecab-ko`)

        - dataset: N/A
        - metric: N/A

    japanese (`mecab-ipadic`)

        - dataset: N/A
        - metric: N/A

    English (`nltk`)

        - dataset: N/A
        - metric: N/A

    Chinese (`jieba`)

        - dataset: N/A
        - metric: N/A

    Args:
        sent (str): input sentence to be tagged

    Returns:
        List[Tuple[str, str]]: list of token and its corresponding pos tag tuple

    Examples:
        >>> pos = Pororo(task="pos", lang="ko")
        >>> pos("안녕하세요. 제 이름은 카터입니다.")
        [('안녕', 'NNG'), ('하', 'XSV'), ('시', 'EP'), ('어요', 'EF'), ('.', 'SF'), (' ', 'SPACE'),
         ('저', 'NP'), ('의', 'JKG'), (' ', 'SPACE'), ('이름', 'NNG'), ('은', 'JX'), (' ', 'SPACE'),
         ('카터', 'NNP'), ('이', 'VCP'), ('ᄇ니다', 'EF'), ('.', 'SF')]
        >>> pos = Pororo("pos", lang="ja")
        >>> pos("日本語でペラペラではないです")
        [('日本語', '名詞'), ('で', '助詞'), ('ペラペラ', '副詞'), ('で', '助動詞'),
         ('は', '助詞'), ('ない', '助動詞'), ('です', '助動詞')]
        >>> pos = Pororo("pos", lang="en")
        >>> pos("The striped bats are hanging, on their feet for best.")
        [('The', 'DT'), (' ', 'SPACE'), ('striped', 'JJ'), (' ', 'SPACE'), ('bats', 'NNS'),
         (' ', 'SPACE'), ('are', 'VBP'), (' ', 'SPACE'), ('hanging', 'VBG'), (',', ','),
         (' ', 'SPACE'), ('on', 'IN'), (' ', 'SPACE'), ('their', 'PRP$'), (' ', 'SPACE'),
         ('feet', 'NNS'), (' ', 'SPACE'), ('for', 'IN'), (' ', 'SPACE'), ('best', 'JJS'), ('.', '.')]
        >>> pos = Pororo("pos", lang="zh")
        >>> pos("乒乓球拍卖完了")
        [('乒乓球', 'n'), ('拍卖', 'v'), ('完', 'v'), ('了', 'ul')]

    """

    def __init__(self, task: str, lang: str, model: Optional[str]):
        super().__init__(task, lang, model)

    @staticmethod
    def get_available_langs():
        return ["en", "ko", "ja", "zh"]

    @staticmethod
    def get_available_models():
        return {
            "en": ["nltk"],
            "ko": ["mecab-ko"],
            "ja": ["mecab-ipadic"],
            "zh": ["jieba"],
        }

    def load(self, device: str):
        """
        Load user-selected task-specific model

        Args:
            device (str): device information

        Returns:
            object: User-selected task-specific model

        """
        if self.config.n_model == "nltk":
            import nltk

            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt")

            try:
                nltk.data.find("taggers/averaged_perceptron_tagger")
            except LookupError:
                nltk.download("averaged_perceptron_tagger")
            return PororoNLTKPosTagger(nltk, self.config)

        if self.config.n_model == "mecab-ko":
            try:
                import mecab
            except ModuleNotFoundError as error:
                raise error.__class__(
                    "Please install python-mecab-ko with: `pip install python-mecab-ko`"
                )
            model = mecab.MeCab()
            return PororoMecabPos(model, self.config)

        if self.config.n_model == "mecab-ipadic":
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
            dic_dir = ipadic.DICDIR
            mecabrc = os.path.join(dic_dir, "mecabrc")
            mecab_option = "-d {} -r {} ".format(
                dic_dir,
                mecabrc,
            )
            model = fugashi.GenericTagger(mecab_option)
            return PororoMecabJap(model, self.config)

        if self.config.n_model == "jieba":
            try:
                import jieba  # noqa
            except ModuleNotFoundError as error:
                raise error.__class__(
                    "Please install jieba with: `pip install jieba`")
            import jieba.posseg as jieba_pos

            model = jieba_pos
            return PororoJieba(model, self.config)


class PororoMecabPos(PororoSimpleBase):

    def __init__(self, model, config):
        super().__init__(config)
        self._model = model

    def _postprocess(self, unit: str) -> Tuple[str, str]:
        """
        Examples:
            >>> parse('나\tNP,*,F,나,*,*,*,*')
            나/NP
            >>> parse('산다\tVV+EC,*,F,산다,Inflect,VV,EC,사/VV/*+ᆫ다/EC/*')
            사/VV+ᆫ다/EC
            >>> parse("', '\tSC,*,*,*,*,*,*,*")
            ,/SC

        """
        # Should split line with tap since comma is frequently used in input sentence
        morph = unit[0]
        features = unit[1]
        pos = features.pos
        analysis = features.expression

        if analysis and ("+" in analysis):
            if "*" in analysis:
                token = [
                    morph.rsplit("/", 1)[0] for morph in analysis.split("+")
                ]
                token = [(t.split("/")[0], t.split("/")[1]) for t in token]
            else:
                analysis = (analysis.replace("+/", "[PLUS]/").replace(
                    "+", "[SEP]").replace("[PLUS]", "+"))
                token = [(pair.split("/")[0], pair.split("/")[1])
                         for pair in analysis.split("[SEP]")]
        else:
            token = (morph, pos)

        return morph, token

    def stringfy(self, result: List[Tuple[str, str]]) -> str:
        res_str = ""
        for pair in result:
            if pair[1] == "SPACE":
                res_str = res_str[:-1]
                res_str += " "
            else:
                res_str += f"{pair[0]}/{pair[1]}+"
        return res_str[:-1]

    def predict(
        self,
        sent: str,
        **kwargs,
    ) -> Union[Tuple[str, str], str]:
        """
        Conduct Part-of-Speech tagging using mecab-ko

        Args:
            sent (str): input sentence to be tagged
            return_surface (bool): whether to return surface
            return_string (bool): whether to return value as a string

        Returns:
            List[Tuple[str, str]]: list of token and its corresponding pos tag tuple

        """
        return_surface = kwargs.get("return_surface", False)
        return_string = kwargs.get("return_string", False)

        sent = sent.strip()
        sent_ptr = 0
        results = []

        if return_surface:
            analyzed = self._model.pos(sent)
        else:
            analyzed = self._model.parse(sent)

        for unit in analyzed:
            if not return_surface:
                morph, token = self._postprocess(unit)
            else:
                token = unit
                morph = unit[0]
            if sent[sent_ptr] == " ":
                # Move sent pointer to whitespace token to reserve whitespace
                # cf. to prevent double white-space, we move pointer to next eojeol
                while sent[sent_ptr] == " ":
                    sent_ptr += 1
                results.append((" ", "SPACE"))
            if isinstance(token, tuple):
                results.append(token)
            elif isinstance(token, list):
                results.extend(token)
            sent_ptr += len(morph)

        if return_string:
            return self.stringfy(results)

        return results


class PororoMecabJap(PororoSimpleBase):

    def __init__(self, model, config):
        super().__init__(config)
        self._model = model

    def predict(self, sent: str, **kwargs):
        """
        Conduct Part-of-Speech tagging using mecab and ipadic modules

        Args:
            sent (str): input sentence to be tagged

        Returns:
            List[Tuple[str, str]]: list of token and its corresponding pos tag tuple

        """
        mecab_output = self._model.parse(sent)

        pairs = list()
        for line in mecab_output.split("\n"):
            if line == "EOS":
                break
            token, tag = line.split("\t")
            tags = tag.split(",")
            pairs.append((token, tags[0]))
        return pairs


class PororoJieba(PororoSimpleBase):

    def __init__(self, model, config):
        super().__init__(config)
        self._model = model

    def predict(self, sent: str, **kwargs):
        """
        Conduct Part-of-Speech tagging using jieba modules

        Args:
            sent (str): input sentence to be tagged

        Returns:
            List[Tuple[str, str]]: list of token and its corresponding pos tag tuple

        """
        jieba_output = self._model.cut(sent)
        return [(word.word, word.flag) for word in list(jieba_output)]


class PororoNLTKPosTagger(PororoSimpleBase):

    def __init__(self, model, config):
        super().__init__(config)
        self._model = model

    def _clean(self, sent: str):
        """
        Cleanse input sentence

        Args:
            sent (str): input sentence to be cleansed

        Returns:
            str: cleansed output sentence

        """
        sent = sent.strip()
        sent = re.sub("\s", " ", sent)
        sent = re.sub(" +", " ", sent)
        return sent

    def _align(self, sent: str, tokens: List[Tuple[str, str]]):
        """
        Align sentence with tagged token pairs

        Args:
            sent (str): original input sentence
            tokens (List[Tuple[str, str]]): list of token and pos tag pair tuple

        Returns:
            List[Tuple[str, str]]: list of aligned token and pos tag pair tuple

        Examples:
            >>> sent = The striped bats are hanging, on their feet for best.
            >>> tokens = [('The', 'DT'), ('striped', 'JJ'), ('bats', 'NNS'), ('are', 'VBP'), ('hanging', 'VBG'), (',', ','), ('on', 'IN'), ('their', 'PRP$'), ('feet', 'NNS'), ('for', 'IN'), ('best', 'JJS'), ('.', '.')]
            >>> align(sent, tokens)
            [('The', 'DT'), (' ', 'SPACE'), ('striped', 'JJ'), (' ', 'SPACE'), ('bats', 'NNS'),
             (' ', 'SPACE'), ('are', 'VBP'), (' ', 'SPACE'), ('hanging', 'VBG'), (',', ','),
             (' ', 'SPACE'), ('on', 'IN'), (' ', 'SPACE'), ('their', 'PRP$'), (' ', 'SPACE'),
             ('feet', 'NNS'), (' ', 'SPACE'), ('for', 'IN'), (' ', 'SPACE'), ('best', 'JJS'), ('.', '.')]

        """
        result = list()
        while True:
            token = tokens.pop(0)
            word = token[0]

            # correct strange behaviors of nltk `word_tokenize`
            # https://github.com/nltk/nltk/issues/1630
            if (word in ("``", "''")) and (sent[0] == '"'):
                word = '"'
                token = ('"', '"')
            if (word == "...") and (sent[0] == "…"):  # ellipsis
                word = "…"
                token = ("…", "…")

            if sent.startswith(f"{word} "):
                sent = sent[len(f"{word} "):]
                result.append(token)
                result.append((" ", "SPACE"))
            elif sent.startswith(word):
                sent = sent[len(word):]
                result.append(token)
            else:
                raise ValueError(f"CANNOT align the {token} to {sent}")

            if not tokens:
                break
        return result

    def predict(self, sent: str, **kwargs):
        """
        Conduct Part-of-Speech tagging using NLTK modules

        Args:
            sent (str): input sentence to be tagged

        Returns:
            List[Tuple[str, str]]: list of token and its corresponding pos tag tuple

        """
        words = self._model.word_tokenize(self._clean(sent))
        pos_tags = self._model.pos_tag(words)
        return self._align(sent, pos_tags)
