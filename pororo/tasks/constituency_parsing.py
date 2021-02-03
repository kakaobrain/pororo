"""Constituency Parsing related modeling class"""

import re
from typing import List, Optional, Tuple

from lxml import etree

from pororo.tasks.utils.base import PororoFactoryBase, PororoTaskBase
from pororo.tasks.utils.download_utils import download_or_load


class PororoConstFactory(PororoFactoryBase):
    """
    Constituency parsing using Transformer model

    English (`transformer.base.en.const`)

        - dataset: OntoNotes 5.0
        - metric: TBU

    Korean (`transformer.base.en.const`)

        - dataset: Sejong Corpus
        - metric: TBU

    Chinese (`transformer.base.zh.const`)

        - dataset: OntoNotes 5.0
        - metric: TBU

    Args:
        text (str): input text
        beam (int): size of beam search
        pos (bool): contains PoS tagging or not

    Returns:
        result: result of constituency parsing

    Examples:
        >>> const = Pororo(task="const", lang="en")
        >>> const("I love this place")
        <TOP>
            <S>
                <NP>I</NP>
                <VP>
                    love
                    <NP>this place</NP>
                </VP>
            </S>
        </TOP>
        >>> const = Pororo(task="const", lang="zh")
        >>> const("我喜欢饼干")
        <TOP>
            <IP>
                <NP>我</NP>
                <VP>
                    喜欢
                    <NP>饼干</NP>
                </VP>
            </IP>
        </TOP>
        >>> const = Pororo(task="const", lang="ko")
        >>> const("미국에서도 같은 우려가 나오고 있다.")
        <S>
          <NP_AJT>미국/NNP+에서/JKB+도/JX</NP_AJT>
            <S>
              <NP_SBJ>
                <VP_MOD>같/VA+은/ETM</VP_MOD>
                <NP_SBJ>우려/NNG+가/JKS</NP_SBJ>
              </NP_SBJ>
              <VP>
                <VP>나오/VV+고/EC</VP>
                <VP>있/VX+다/EF+./SF</VP>
              </VP>
            </S>
        </S>

    """

    def __init__(self, task: str, lang: str, model: Optional[str]):
        super().__init__(task, lang, model)

    @staticmethod
    def get_available_langs():
        return ["en", "ko", "zh"]

    @staticmethod
    def get_available_models():
        return {
            "en": ["transformer.base.en.const"],
            "ko": ["transformer.base.ko.const"],
            "zh": ["transformer.base.zh.const"],
        }

    def load(self, device: str):
        """
        Load user-selected task-specific model

        Args:
            device (str): device information

        Returns:
            object: User-selected task-specific model

        """
        if "transformer" in self.config.n_model:
            from fairseq.models.transformer import TransformerModel

            from pororo.tasks import PororoPosFactory

            load_dict = download_or_load(
                f"transformer/{self.config.n_model}",
                self.config.lang,
            )

            model = (TransformerModel.from_pretrained(
                model_name_or_path=load_dict.path,
                checkpoint_file=f"{self.config.n_model}.pt",
                data_name_or_path=load_dict.dict_path,
                source_lang=load_dict.src_dict,
                target_lang=load_dict.tgt_dict,
            ).eval().to(device))

            if self.config.lang == "ko":
                tagger = PororoPosFactory(
                    task="pos",
                    model="mecab-ko",
                    lang=self.config.lang,
                ).load(device)
                return PororoTransConstKo(model, tagger, self.config)

            if self.config.lang == "en":
                tagger = PororoPosFactory(
                    task="pos",
                    model="nltk",
                    lang=self.config.lang,
                ).load(device)
                return PororoTransConstEn(model, tagger, self.config)

            if self.config.lang == "zh":
                tagger = PororoPosFactory(
                    task="pos",
                    model="jieba",
                    lang=self.config.lang,
                ).load(device)
                return PororoTransConstZh(model, tagger, self.config)


class PororoConstBase(PororoTaskBase):
    """Constituency Parsing base class containinig various methods related to Const. Parsing"""

    def _fix_tree(self, output: str):
        """
        Fix tree when XML conversion is not conducted

        Args:
            output (str): string to fix

        Returns:
            text: fixed tree string

        """
        tag_ptn = "[A-Z][A-Z_]*"
        output = re.sub("\s", "", output)
        xml = re.sub(f"<({tag_ptn})>", r"［\1 ", output)
        xml = re.sub(f"</{tag_ptn}>", r"］ ", xml)

        def _convert_to_xml(text):
            for _ in range(max(text.count("［"), text.count("］"))):
                text = re.sub(
                    f"(?s)［({tag_ptn})([^［］]+?)］",
                    r"<\1>\2 </\1>",
                    text,
                )
            return text

        xml = _convert_to_xml(xml)
        xml = re.sub(f"［{tag_ptn}", "", xml)
        xml = re.sub(f"{tag_ptn}］", "", xml)
        xml = re.sub("[［］\s]", "", xml)
        return xml

    def _prettify(self, output: str):
        """
        Prettify model result using XML tree

        Args:
            output (str): string to make tree

        Returns:
            pretty: tree style output

        """
        output = re.sub("> +", ">", output)
        output = re.sub(" +<", "<", output)
        output = re.sub(
            "(<[A-Za-z_\d]+>) *([^< ]+) *(<[^/])",
            r"\1<temp>\2</temp>\3",
            output,
        )
        output = re.sub(
            "(</[A-Za-z_\d]+>) *([^< ]+) *(</)",
            r"\1<temp>\2</temp>\3",
            output,
        )
        try:
            root = etree.fromstring(output)
        except:
            root = etree.fromstring(self._fix_tree(output))
        tree = etree.ElementTree(root)
        pretty = etree.tostring(tree, pretty_print=True, encoding="unicode")
        pretty = pretty.replace("<temp>", "").replace("</temp>", "")
        return pretty.replace("  ", "\t")

    def __call__(
        self,
        text: str,
        beam: int = 5,
        pos: bool = False,
        **kwargs,
    ):
        """
        Conduct constituency parsing

        Args:
            text (str): input text
            beam (int): size of beam search
            pos (bool): contains PoS tagging or not

        Returns:
            result: result of constituency parsing

        """
        assert isinstance(text, str), "Input text should be string type"

        text = self._normalize(text)

        return self.predict(text, beam, pos, **kwargs)


class PororoTransConstKo(PororoConstBase):

    def __init__(self, model, tagger, config):
        super().__init__(config)
        self._model = model
        self._tagger = tagger

    def _postprocess(
        self,
        result: List[str],
        eojeols: List[str],
        poses: List[str],
    ):
        """
        Postprocess method to make XML format

        Args:
            result (List[str]): constituency parsing result
            eojeols (List): list of eojeol
            poses (List): list of pos tag

        Returns:
            str: result of postprocess

        """
        token_indices = []
        temp_group = []
        for i, res in enumerate(result):
            if ("<" in res) or (">" in res):
                continue
            if not temp_group:
                temp_group.append(i)
            else:
                if i == (temp_group[-1] + 1):
                    temp_group.append(i)
                else:
                    token_indices.append(temp_group)
                    temp_group = [i]
        token_indices.append(temp_group)

        lucrative = 0
        for i, li_index in enumerate(token_indices):
            if poses:
                eojeol = eojeols[i].split("+")
                pos = poses[i].split("+")
                tagged = []
                for e, p in zip(eojeol, pos):
                    tagged.append(f"{e}/{p}")
                result[li_index[0] - lucrative:li_index[-1] + 1 -
                       lucrative] = ["+".join(tagged)]
            else:
                result[li_index[0] - lucrative:li_index[-1] + 1 -
                       lucrative] = [eojeols[i]]
            lucrative += len(li_index) - 1

        return result

    def _check_sanity(self, cands: List[str], n_space: int):
        """
        Check sanity for valid xml structure

        Args:
            cands (List[str]): candidates
            n_space (int): number of space

        Returns:
            return valid or not

        """
        for cand in cands:
            # Count the number of space special character
            if cand.count("▁") != n_space:
                continue
            # Check whether candidate XML is valid
            try:
                etree.fromstring(cand)
                return cand
            except:
                continue
        return False

    def predict(
        self,
        text: str,
        beam: int = 5,
        pos: bool = False,
        **kwargs,
    ):
        """
        Conduct constituency parsing

        Args:
            text (str): input text
            beam (int): size of beam search
            pos (bool): contains PoS tagging or not

        Returns:
            result of constituency parsing

        """
        eojeols = self._tagger(text)
        n_space = len([m for m in eojeols if m[1] == "SPACE"])
        pairs = self._tagger(text, return_string=False)
        src = " ".join(
            [pair[1] if pair[1] != "SPACE" else "▁" for pair in pairs])

        outputs = self._model.translate(
            src,
            beam=beam,
            max_len_a=1,
            max_len_b=50,
        )
        result = self._check_sanity([outputs], n_space)

        if not result:
            return f"<ERROR> {text} </ERROR>"
        result = [res for res in result.split() if res != "▁"]

        words = []
        poses = []
        tmp_word = ""
        tmp_pos = ""
        for eojeol in eojeols:
            if eojeol[1] != "SPACE":
                tmp_word += f"{eojeol[0]}+"
                tmp_pos += f"{eojeol[1]}+"
            else:
                words.append(tmp_word[:-1])
                poses.append(tmp_pos[:-1])
                tmp_word = ""
                tmp_pos = ""

        words.append(tmp_word[:-1])
        poses.append(tmp_pos[:-1])

        if not pos:
            poses = None

        result = " ".join(self._postprocess(result, words, poses))
        return self._prettify(result).strip()


class PororoTransConstEn(PororoConstBase):

    def __init__(self, model, tagger, config):
        super().__init__(config)
        self._model = model
        self._tagger = tagger

    def _check_sanity(self, tags: List[str], n_words: int):
        """
        Check sanity for valid xml structure

        Args:
            tags (List[str]): list of tags
            n_words (int): number of words

        Returns:
            return valid or not

        """
        n_out = 0

        for tag in tags:
            if ("<" not in tag) and (">" not in tag):
                n_out += 1

        return n_out == n_words

    def _preprocess(self, tagged: List[Tuple]) -> str:
        """
        Preprocess input sentence to replace whitespace token with whitespace

        Args:
            tagged (List[str]): list of tagges

        Returns:
            preprocessed sentence, original input

        """
        ori = " ".join([tag[0] for tag in tagged if tag[1] != "SPACE"])
        sent = " ".join([tag[1] for tag in tagged if tag[1] != "SPACE"])
        sent = sent.replace("-LRB-", "(")
        sent = sent.replace("-RRB-", ")")
        return sent, ori

    def _postprocess(self, tags: List[str], words: List[str], pos: List[str]):
        """
        Postprocess result of parsing

        Args:
            tags (List[str]): list of parsing tag
            words (List[str]): list of word
            pos (List[str]): list of PoS tag

        Returns:
            postprocessed result string

        """
        result = list()

        i = 0
        for tag in tags:
            if ("<" not in tag) and (">" not in tag):
                if pos:
                    result.append(f"{words[i]}/{pos[i]}")
                else:
                    result.append(words[i])
                i += 1
            else:
                result.append(tag)

        return " ".join(result)

    def predict(
        self,
        text: str,
        beam: int = 5,
        pos: bool = False,
        **kwargs,
    ):
        """
        Conduct constituency parsing

        Args:
            text (str): input sentence
            beam (int): size of beam search
            pos (bool): contains PoS tagging or not

        Returns:
            result of constituency parsing

        """
        tags, ori = self._preprocess(self._tagger(text))
        n_words = len(tags.split())
        outputs = self._model.translate(
            tags,
            beam=beam,
            max_len_a=1,
            max_len_b=50,
        )
        result = self._check_sanity(outputs.split(), n_words)

        if not result:
            return f"<ERROR> {text} </ERROR>"

        poses = None
        if pos:
            poses = tags.split()

        outputs = self._postprocess(outputs.split(), ori.split(), poses)
        return self._prettify(outputs).strip()


class PororoTransConstZh(PororoConstBase):

    def __init__(self, model, tagger, config):
        super().__init__(config)
        self._model = model
        self._tagger = tagger
        self._map = {
            "a": "ADJ",
            "ad": "ADJ",
            "ag": "ADJ",
            "an": "ADJ",
            "b": "NOUN",
            "c": "CONJ",
            "d": "ADV",
            "df": "ADV",
            "dg": "ADV",
            "e": "INTJ",
            "f": "NOUN",
            "g": "MORPHEME",
            "h": "PREFIX",
            "i": "IDIOM",
            "j": "NOUN",
            "k": "SUFFIX",
            "l": "IDIOM",
            "m": "NUM",
            "mg": "NUM",
            "mq": "NUM",
            "n": "NOUN",
            "ng": "NOUN",
            "nr": "NOUN",
            "nrfg": "NOUN",
            "nrt": "NOUN",
            "ns": "NOUN",
            "nt": "NOUN",
            "nz": "NOUN",
            "o": "ONOM",
            "p": "PREP",
            "q": "CLASSIFIER",
            "r": "PRON",
            "rg": "PRON",
            "rr": "PRON",
            "rz": "PRON",
            "s": "NOUN",
            "t": "NOUN",
            "tg": "NOUN",
            "u": "PART",
            "ud": "PART",
            "ug": "PART",
            "uj": "PART",
            "ul": "PART",
            "uv": "PART",
            "uz": "PART",
            "v": "VERB",
            "vd": "VERB",
            "vg": "VERB",
            "vi": "VERB",
            "vn": "VERB",
            "vq": "VERB",
            "x": "X",
            "y": "PART",
            "z": "ADJ",
            "zg": "ADJ",
            "eng": "X",
        }

    def _check_sanity(self, tags: List[str], n_words: int):
        """
        Check sanity for valid xml structure

        Args:
            tags (List[str]): list of tag
            n_words (int): number of word

        Returns:
            return valid or not

        """
        n_out = 0

        for tag in tags:
            if ("<" not in tag) and (">" not in tag):
                n_out += 1

        return n_out == n_words

    def _preprocess(self, tagged: List[Tuple]) -> Tuple:
        """
        Preprocess input sentence to replace whitespace token with whitespace

        Args:
            tagged (List[Tuple]): list of tagged tuple

        Returns:
            result of preprocess

        """
        ori = " ".join([tag[0] for tag in tagged])
        tags = [tag[1] for tag in tagged]
        # Mapping into general tagset
        tags = [self._map[tag] if tag in self._map else "X" for tag in tags]
        return " ".join(tags), ori

    def _postprocess(
        self,
        tags: List[str],
        words: List[str],
        pos: bool = False,
    ):
        """
        Postprocess result of parsing

        Args:
            tags (List[str]): list of parsing tag
            words (List[str]): list of word
            pos (List[str]): list of PoS tag

        Returns:
            postprocessed result string

        """
        result = list()

        i = 0
        for tag in tags:
            if ("<" not in tag) and (">" not in tag):
                if pos:
                    result.append(f"{words[i]}/{pos[i]}")
                else:
                    result.append(words[i])
                i += 1
            else:
                result.append(tag)

        return " ".join(result)

    def predict(
        self,
        text: str,
        beam: int = 5,
        pos: bool = False,
        **kwargs,
    ):
        """
        Conduct constituency parsing

        Args:
            text (str): input sentence
            beam (int): size of beam search
            pos (bool): contains PoS tagging or not

        Returns:
            result of constituency parsing

        """
        tags, ori = self._preprocess(self._tagger(text))
        n_words = len(tags.split())

        outputs = self._model.translate(
            tags,
            beam=beam,
            max_len_a=1,
            max_len_b=50,
        )
        result = self._check_sanity(outputs.split(), n_words)

        if not result:
            return f"<ERROR> {text} </ERROR>"

        poses = None
        if pos:
            poses = tags.split()

        outputs = self._postprocess(outputs.split(), ori.split(), poses)
        return self._prettify(outputs).strip()
