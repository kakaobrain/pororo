"""Named Entity Recognition related modeling class"""

import re
from collections import defaultdict
from typing import List, Optional, Tuple

from pororo.tasks.utils.base import PororoFactoryBase, PororoSimpleBase
from pororo.tasks.utils.download_utils import download_or_load


class PororoNerFactory(PororoFactoryBase):
    """
    Conduct named entity recognition

    English (`roberta.base.en.ner`)

        - dataset: OntoNotes 5.0
        - metric: F1 (91.63)

    Korean (`charbert.base.ko.ner`)

        - dataset: https://corpus.korean.go.kr/ 개체명 분석 말뭉치
        - metric: F1 (89.63)

    Japanese (`jaberta.base.ja.ner`)

        - dataset: Kyoto University Web Document Leads Corpus
        - metric: F1 (76.74)
        - ref: https://github.com/ku-nlp/KWDLC

    Chinese (`zhberta.base.zh.ner`)

        - dataset: OntoNotes 5.0
        - metric: F1 (79.06)

    Args:
        sent: (str) sentence to be sequence labeled

    Returns:
        List[Tuple[str, str]]: token and its predicted tag tuple list

    Examples:
        >>> ner = Pororo(task="ner", lang="en)
        >>> ner("It was in midfield where Arsenal took control of the game, and that was mainly down to Thomas Partey and Mohamed Elneny.")
        [('It', 'O'), ('was', 'O'), ('in', 'O'), ('midfield', 'O'), ('where', 'O'), ('Arsenal', 'ORG'), ('took', 'O'), ('control', 'O'), ('of', 'O'), ('the', 'O'), ('game', 'O'), (',', 'O'), ('and', 'O'), ('that', 'O'), ('was', 'O'), ('mainly', 'O'), ('down', 'O'), ('to', 'O'), ('Thomas Partey', 'PERSON'), ('and', 'O'), ('Mohamed Elneny', 'PERSON'), ('.', 'O')]
        >>> ner = Pororo(task="ner", lang="ko")
        >>> ner("손흥민은 28세의 183 센티미터, 77 킬로그램이며, 현재 주급은 약 3억 원이다.")
        [('손흥민', 'PERSON'), ('은', 'O'), (' ', 'O'), ('28세', 'QUANTITY'), ('의', 'O'), (' ', 'O'), ('183 센티미터', 'QUANTITY'), (',', 'O'), (' ', 'O'), ('77 킬로그램', 'QUANTITY'), ('이며,', 'O'), (' ', 'O'), ('현재', 'O'), (' ', 'O'), ('주급은', 'O'), (' ', 'O'), ('약 3억 원', 'QUANTITY'), ('이다.', 'O')]
        >>> # `apply_wsd` : for korean, you can use Word Sense Disambiguation module to get more specific tag
        >>> ner("손흥민은 28세의 183 센티미터, 77 킬로그램이며, 현재 주급은 약 3억 원이다.", apply_wsd=True)
        [('손흥민', 'PERSON'), ('은', 'O'), (' ', 'O'), ('28세', 'AGE'), ('의', 'O'), (' ', 'O'), ('183 센티미터', 'LENGTH/DISTANCE'), (',', 'O'), (' ', 'O'), ('77 킬로그램', 'WEIGHT'), ('이며,', 'O'), (' ', 'O'), ('현재', 'O'), (' ', 'O'), ('주급은', 'O'), (' ', 'O'), ('약 3억 원', 'MONEY'), ('이다.', 'O')]
        >>> ner = Pororo(task="ner", lang="zh")
        >>> ner("毛泽东（1893年12月26日－1976年9月9日），字润之，湖南湘潭人。中华民国大陆时期、中国共产党和中华人民共和国的重要政治家、经济家、军事家、战略家、外交家和诗人。")
        [('毛泽东', 'PERSON'), ('（', 'O'), ('1893年12月26日－1976年9月9日', 'DATE'), ('）', 'O'), ('，', 'O'), ('字润之', 'O'), ('，', 'O'), ('湖南', 'GPE'), ('湘潭', 'GPE'), ('人', 'O'), ('。', 'O'), ('中华民国大陆时期', 'GPE'), ('、', 'O'), ('中国共产党', 'ORG'), ('和', 'O'), ('中华人民共和国', 'GPE'), ('的', 'O'), ('重', 'O'), ('要', 'O'), ('政', 'O'), ('治', 'O'), ('家', 'O'), ('、', 'O'), ('经', 'O'), ('济', 'O'), ('家', 'O'), ('、', 'O'), ('军', 'O'), ('事', 'O'), ('家', 'O'), ('、', 'O'), ('战', 'O'), ('略', 'O'), ('家', 'O'), ('、', 'O'), ('外', 'O'), ('交', 'O'), ('家', 'O'), ('和', 'O'), ('诗', 'O'), ('人', 'O'), ('。', 'O')]
        >>> ner = Pororo(task="ner", lang="ja")
        >>> ner("豊臣 秀吉、または羽柴 秀吉は、戦国時代から安土桃山時代にかけての武将、大名。天下人、武家関白、太閤。三英傑の一人。")
        [('豊臣秀吉', 'PERSON'), ('、', 'O'), ('または', 'O'), ('羽柴秀吉', 'PERSON'), ('は', 'O'), ('、', 'O'), ('戦国時代', 'DATE'), ('から', 'O'), ('安土桃山時代', 'DATE'), ('にかけて', 'O'), ('の', 'O'), ('武将', 'O'), ('、', 'O'), ('大名', 'O'), ('。', 'O'), ('天下', 'O'), ('人', 'O'), ('、', 'O'), ('武家', 'O'), ('関白', 'O'), ('、太閤', 'O'), ('。', 'O'), ('三', 'O'), ('英', 'O'), ('傑', 'O'), ('の', 'O'), ('一', 'O'), ('人', 'O'), ('。', 'O')]

    """

    def __init__(self, task: str, lang: str, model: Optional[str]):
        super().__init__(task, lang, model)

    @staticmethod
    def get_available_langs():
        return ["en", "ko", "zh", "ja"]

    @staticmethod
    def get_available_models():
        return {
            "en": ["roberta.base.en.ner"],
            "ko": ["charbert.base.ko.ner"],
            "zh": ["zhberta.base.zh.ner"],
            "ja": ["jaberta.base.ja.ner"],
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
            return PororoBertNerEn(model, self.config)

        if "charbert" in self.config.n_model:
            from pororo.models.brainbert import CharBrainRobertaModel
            from pororo.tasks.tokenization import PororoTokenizationFactory

            model = (CharBrainRobertaModel.load_model(
                f"bert/{self.config.n_model}",
                self.config.lang,
            ).eval().to(device))

            sent_tokenizer = PororoTokenizationFactory(
                task="tokenization",
                model="sent_ko",
                lang=self.config.lang,
            ).load(device)

            f_wsd_dict = open(
                download_or_load(
                    f"misc/wiki.{self.config.lang}.items",
                    self.config.lang,
                ),
                "r",
            )
            wsd_dict = defaultdict(dict)
            for line in f_wsd_dict.readlines():
                origin, target, word = line.strip().split("\t")
                wsd_dict[origin][word] = target

            return PororoBertCharNer(
                model,
                sent_tokenizer,
                wsd_dict,
                device,
                self.config,
            )

        if "zhberta" in self.config.n_model:
            from pororo.models.brainbert import ZhbertaModel

            model = (ZhbertaModel.load_model(
                f"bert/{self.config.n_model}",
                self.config.lang,
            ).eval().to(device))
            return PororoBertNerZh(model, self.config)

        if "jaberta" in self.config.n_model:
            from pororo.models.brainbert import JabertaModel

            model = (JabertaModel.load_model(
                f"bert/{self.config.n_model}",
                self.config.lang,
            ).eval().to(device))
            return PororoBertNerJa(model, self.config)


class PororoBertNerEn(PororoSimpleBase):

    def __init__(self, model, config):
        super().__init__(config)
        self._model = model

    def _postprocess(self, tags: List[str]):
        """
        Postprocess NER tags to concatenate BIO

        Args:
            tags (List[str]): inferenced tag list

        Returns:
            List[str]: postprocessed BIO scheme tag list

        """

        def _remove_tail(tag):
            if "-" in tag:
                tag = tag[2:]
            return tag

        result = list()

        word = tags[0][0]
        tag = tags[0][1]

        for pair in tags[1:]:
            token, label = pair
            if "I" in label:
                word += token
            else:
                word = word.strip()
                if word.endswith("."):
                    result.append((word[:-1], _remove_tail(tag)))
                    result.append((".", "O"))
                else:
                    result.append((word, _remove_tail(tag)))
                word = token
                tag = label

        word = word.strip()
        if word.endswith("."):
            result.append((word[:-1], _remove_tail(tag)))
            result.append((".", "O"))
        else:
            result.append((word, _remove_tail(tag)))
        return [pair for pair in result if pair[0] != ""]

    def predict(self, sent: str, **kwargs):
        """
        Conduct named entity recognition with english RoBERTa

        Args:
            sent: (str) sentence to be sequence labeled

        Returns:
            List[Tuple[str, str]]: token and its predicted tag tuple list

        """
        return self._postprocess(self._model.predict_tags(sent))


class PororoBertCharNer(PororoSimpleBase):

    def __init__(
        self,
        model,
        sent_tokenizer,
        wsd_dict,
        device,
        config,
    ):
        super().__init__(config)
        self._model = model
        self._sent_tokenizer = sent_tokenizer
        self._tag = {
            "PS": "PERSON",
            "LC": "LOCATION",
            "OG": "ORGANIZATION",
            "AF": "ARTIFACT",
            "DT": "DATE",
            "TI": "TIME",
            "CV": "CIVILIZATION",
            "AM": "ANIMAL",
            "PT": "PLANT",
            "QT": "QUANTITY",
            "FD": "STUDY_FIELD",
            "TR": "THEORY",
            "EV": "EVENT",
            "MT": "MATERIAL",
            "TM": "TERM",
        }
        self._device = device
        self._wsd_dict = wsd_dict

        self._wsd = None
        self._cls2cat = None
        self._quant2cat = None
        self._term2cat = None

    def _template_match(self, text, expression2category):
        """
        Apply template match using regular expression

        Args:
            text (str): text to be searched
            expression2category (dict): regular expression dict

        Returns:
            str: regex matched category

        """
        for expression, category in expression2category.items():
            if re.search(expression, text) is not None:
                return category

    def apply_dict(self, tags: List[Tuple[str, str]]):
        """
        Apply pre-defined dictionary to get detail tag info

        Args:
            tags (List[Tuple[str, str]]): inference word-tag pair result

        Returns:
            List[Tuple[str, str]]: dict-applied result

        """
        result = []
        for pair in tags:
            word, tag = pair
            if (tag in self._wsd_dict.keys()) and (word in self._wsd_dict[tag]):
                result.append((word, self._wsd_dict[tag][word].upper()))
            else:
                result.append(pair)
        return result

    def _apply_wsd(self, tags: List[Tuple[str, str]]):
        """
        Apply Word Sense Disambiguation to get detail tag info

        Args:
            tags (List[Tuple[str, str]]): inference word-tag pair result

        Returns:
            List[Tuple[str, str]]: wsd-applied result

        """
        if self._wsd is None:
            from pororo.tasks import PororoWsdFactory
            self._wsd = PororoWsdFactory(
                task="wsd",
                lang="ko",
                model="transformer.large.ko.wsd",
            ).load(self._device)

        if self._cls2cat is None:
            self._cls2cat = dict()
            lines = (open(
                download_or_load(
                    "misc/wsd.cls.txt",
                    self.config.lang,
                ),
                "r",
                encoding="utf8",
            ).read().strip().splitlines())
            for line in lines:
                morph, homonymno, category = line.split()
                classifier = f"{morph}__NNB__{homonymno}"  # bound noun
                self._cls2cat[classifier] = category

        if self._quant2cat is None:
            self._quant2cat = dict()
            self._term2cat = dict()
            lines = (open(
                download_or_load(
                    "misc/re.templates.txt",
                    self.config.lang,
                ),
                "r",
            ).read().strip().splitlines())

            for line in lines:
                category, ner_category, expression = line.split(" ", 2)
                if ner_category == "QUANTITY":
                    self._quant2cat[expression] = category
                elif ner_category == "TERM":
                    self._term2cat[expression] = category

        input_text_with_markers = str()
        target_token_ids = []

        for idx, ner_token in enumerate(tags):
            surface, tag = ner_token
            # as {} will be used as special symbols
            surface = surface.replace("{", "｛")
            surface = surface.replace("}", "｝")

            if tag == "TERM":
                cat = self._template_match(surface, self._term2cat)
                if cat is not None:
                    tags[idx] = (surface, cat)
                input_text_with_markers += surface
            elif tag == "QUANTITY":
                cat = self._template_match(surface, self._quant2cat)
                if cat is not None:
                    tags[idx] = (surface, cat)
                    input_text_with_markers += surface
                else:
                    target_token_ids.append(idx)
                    input_text_with_markers += "{" + surface + "}"
            else:
                input_text_with_markers += surface

        wsd_results = self._wsd(input_text_with_markers)
        action = False
        has_category = False
        categories = []

        for wsd_token in wsd_results:
            morph, tag, homonymno = wsd_token[:3]
            if morph == "{":
                has_category = False
                action = True
            elif morph == "}":
                if has_category is False:
                    categories.append("QUANTITY")  # original category
                has_category = False
                action = False

            if action:
                if homonymno is None:
                    homonymno = "00"

                query = f"{morph}__{tag}__{homonymno}"
                if query in self._cls2cat:
                    category = self._cls2cat[query]
                    categories.append(category)
                    has_category = True
                    action = False

        assert len(target_token_ids) == len(categories)

        for target_token_id, cat in zip(target_token_ids, categories):
            tags[target_token_id] = (tags[target_token_id][0], cat)

        return tags

    def _postprocess(self, tags: List[Tuple[str, str]]):
        """
        Postprocess characted tags to concatenate BIO

        Args:
            tags (List[Tuple[str, str]]): characted token and its corresponding tag tuple list

        Returns:
            List(Tuple[str, str]): postprocessed entity token and its corresponding tag tuple list

        """

        def _remove_tail(tag: str):
            if "-" in tag:
                tag = tag[:-2]
            return tag

        result = list()

        tmp_word = tags[0][0]
        prev_ori_tag = tags[0][1]
        prev_tag = _remove_tail(prev_ori_tag)
        for _, pair in enumerate(tags[1:]):
            char = pair[0]
            ori_tag = pair[1]
            tag = _remove_tail(ori_tag)
            if ("▁" in char) and ("-I" not in ori_tag):
                result.append((tmp_word, prev_tag))
                result.append((" ", "O"))

                tmp_word = char
                prev_tag = tag
                continue

            if (tag == prev_tag) and (("-I" in ori_tag) or "O" in ori_tag):
                tmp_word += char
            elif (tag != prev_tag) and ("-I" in ori_tag) and (tag != "O"):
                tmp_word += char
            else:
                result.append((tmp_word, prev_tag))
                tmp_word = char

            prev_tag = tag
        result.append((tmp_word, prev_tag))

        result = [(pair[0].replace("▁", " ").strip(),
                   pair[1]) if pair[0] != " " else (" ", "O")
                  for pair in result]
        return result

    def predict(
        self,
        text: str,
        **kwargs,
    ):
        """
        Conduct named entity recognition with character BERT

        Args:
            text: (str) sentence to be sequence labeled

        Returns:
            List[Tuple[str, str]]: token and its predicted tag tuple list

        """
        apply_wsd = kwargs.get("apply_wsd", False)
        ignore_labels = kwargs.get("ignore_labels", [])

        texts = text.strip().split("\n")
        result = []
        for text in texts:
            for sent in self._sent_tokenizer(text.strip()):
                res = self._model.predict_tags(sent)
                res = [
                    pair for pair in self._postprocess(res)
                    if pair[1] not in ignore_labels
                ]
                res = [(
                    pair[0],
                    self._tag[pair[1]],
                ) if pair[1] in self._tag else pair for pair in res]
                res = res if not apply_wsd else self._apply_wsd(res)
                result.extend(self.apply_dict(res))
                result.extend([(" ", "O")])
        return result[:-1]


class PororoBertNerZh(PororoSimpleBase):

    def __init__(self, model, config):
        super().__init__(config)
        self._model = model

    def _postprocess(
        self,
        tags: List[str],
    ):
        """
        Postprocess NER tags to concatenate BIO

        Args:
            tags (List[str]): inferenced tag list

        Returns:
            List[str]: postprocessed BIO scheme tag list

        """

        def _remove_tail(tag):
            if "-" in tag:
                tag = tag[2:]
            return tag

        result = list()

        word = tags[0][0]
        tag = tags[0][1]
        for pair in tags[1:]:
            token, label = pair
            if "I" in label:
                word += token
            else:
                word = word.strip()
                result.append((word, _remove_tail(tag)))
                word = token
                tag = label

        word = word.strip()
        result.append((word, _remove_tail(tag)))
        return result

    def predict(self, sent: str, **kwargs):
        """
        Conduct named entity recognition with Chinese RoBERTa

        Args:
            sent: (str) sentence to be sequence labeled

        Returns:
            List[Tuple[str, str]]: token and its predicted tag tuple list

        """
        tags = self._model.predict_tags(sent)
        return self._postprocess(tags)


class PororoBertNerJa(PororoSimpleBase):

    def __init__(self, model, config):
        super().__init__(config)
        self._model = model

    def _postprocess(
        self,
        tags: List[str],
    ):
        """
        Postprocess NER tags to concatenate BIO

        Args:
            tags (List[str]): inferenced tag list

        Returns:
            List[str]: postprocessed BIO scheme tag list

        """

        def _remove_tail(tag):
            if "-" in tag:
                tag = tag[2:]
            return tag

        result = list()

        word = tags[0][0]
        tag = tags[0][1]
        for pair in tags[1:]:
            token, label = pair
            if "I" in label:
                word += token
            else:
                word = word.strip()
                result.append((word.replace("##", ""), _remove_tail(tag)))
                word = token
                tag = label

        word = word.strip()
        result.append((word.replace("##", ""), _remove_tail(tag)))
        return result

    def predict(self, sent: str, **kwargs):
        """
        Conduct named entity recognition with Japanese RoBERTa

        Args:
            sent: (str) sentence to be sequence labeled

        Returns:
            List[Tuple[str, str]]: token and its predicted tag tuple list

        """
        return self._postprocess(self._model.predict_tags(sent))
