"""Tokenization related modeling class"""

import json
import os
import re
from abc import abstractmethod
from typing import List, Optional
from unicodedata import normalize

from kss import split_sentences

from pororo.tasks.utils.base import PororoFactoryBase, PororoSimpleBase
from pororo.tasks.utils.download_utils import download_or_load


class PororoTokenizationFactory(PororoFactoryBase):
    """
    Use the dictionary you want to use to tokenize about the sentence.

    Args:
        sent: (str) sentence to be tokenized

    Returns:
        List[str]: tokenized token list

    Examples:
        >>> tk = Pororo(task="tokenization", lang="ko", model="bpe32k.ko", )
        >>> tk("하늘을 나는 새를 보았다")
        ["_하늘", "을", "_나는", "_새", "를", "_보", "았다"]
        >>> tk = Pororo(task="tokenization", lang="en", model="roberta")
        >>> tk("I love you")
        ['I', 'Ġlove', 'Ġyou']
        >>> tk('''If the values aren’t unique, there is no unique inversion of the dictionary anyway or, with other words, inverting does not make sense.''')
        ['If', 'Ġthe', 'Ġvalues', 'Ġaren', 'âĢ', 'Ļ', 't', 'Ġunique', ',', 'Ġthere', 'Ġis', 'Ġno', 'Ġunique', 'Ġin', 'version', 'Ġof', 'Ġthe', 'Ġdictionary', 'Ġanyway', 'Ġor', ',', 'Ġwith', 'Ġother', 'Ġwords', ',', 'Ġinver', 'ting', 'Ġdoes', 'Ġnot', 'Ġmake', 'Ġsense', '.']

    """

    def __init__(self, task: str, lang: str, model: Optional[str]):
        super().__init__(task, lang, model)

    @staticmethod
    def get_available_langs():
        return ["en", "ko", "ja", "zh"]

    @staticmethod
    def get_available_models():
        return {
            "en": [
                "moses",
                "bpe32k.en",
                "roberta",
                "sent_en",
            ],
            "ko": [
                "bpe4k.ko",
                "bpe8k.ko",
                "bpe16k.ko",
                "bpe32k.ko",
                "bpe64k.ko",
                "unigram4k.ko",
                "unigram8k.ko",
                "unigram16k.ko",
                "unigram32k.ko",
                "unigram64k.ko",
                "jpe4k.ko",
                "jpe8k.ko",
                "jpe16k.ko",
                "jpe32k.ko",
                "jpe64k.ko",
                "mecab.bpe4k.ko",
                "mecab.bpe8k.ko",
                "mecab.bpe16k.ko",
                "mecab.bpe32k.ko",
                "mecab.bpe64k.ko",
                "char",
                "jamo",
                "word",
                "mecab_ko",
                "sent_ko",
            ],
            "ja": [
                "mecab",
                "bpe8k.ja",
                "sent_ja",
            ],
            "zh": [
                "jieba",
                "sent_zh",
            ],
        }

    def load(self, device: str):
        """
        Load user-selected task-specific model

        Args:
            device (str): device information

        Returns:
            object: User-selected task-specific model

        """
        if "sent" in self.config.n_model:
            import nltk

            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt")

            from nltk.tokenize import sent_tokenize

            return PororoSentTokenizer(sent_tokenize, self.config)

        if self.config.n_model == "mecab_ko":
            try:
                import mecab
            except ModuleNotFoundError as error:
                raise error.__class__(
                    "Please install python-mecab-ko with: `pip install python-mecab-ko`"
                )
            model = mecab.MeCab()
            return PororoMecabKoTokenizer(model, self.config)

        if self.config.n_model == "char":
            return PororoCharTokenizer(self.config)

        if self.config.n_model == "jamo":
            return PororoJamoTokenizer(self.config)

        if self.config.n_model == "word":
            return PororoWordTokenizer(self.config)

        if self.config.n_model == "roberta":
            from fairseq.data.encoders.gpt2_bpe import get_encoder

            encoder = download_or_load("misc/encoder.json", self.config.lang)
            vocab = download_or_load("misc/vocab.bpe", self.config.lang)
            model = get_encoder(encoder, vocab)

            with open(encoder, "r") as f_vocab:
                vocab = json.load(f_vocab)
                inv_dict = {v: k for k, v in vocab.items()}

            return PororoRoBERTaTokenizer(model, vocab, inv_dict, self.config)

        if self.config.n_model == "moses":
            try:
                from sacremoses import MosesDetokenizer, MosesTokenizer
            except ModuleNotFoundError as error:
                raise error.__class__(
                    "Please install sacremoses with: `pip install sacremoses`")
            model = MosesTokenizer(lang="en")
            detok = MosesDetokenizer(lang="en")
            return PororoMosesTokenizer(model, detok, self.config)

        if self.config.n_model == "jieba":
            try:
                import jieba
            except ModuleNotFoundError as error:
                raise error.__class__(
                    "Please install jieba with: `pip install jieba`")
            model = jieba.cut
            return PororoJiebaTokenizer(model, self.config)

        if self.config.n_model == "mecab":
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
            return PororoMecabTokenizer(model, self.config)
        else:
            from pororo.tasks.utils.tokenizer import CustomTokenizer

            path = download_or_load(
                f"tokenizers/{self.config.n_model}.zip",
                self.config.lang,
            )

            ext = "json" if "unigram" not in self.config.n_model else "txt"
            merges_filename = (f"{path}/merges.txt" if "unigram"
                               not in self.config.n_model else None)

            model = CustomTokenizer.from_file(
                vocab_filename=f"{path}/vocab.{ext}",
                merges_filename=merges_filename,
                normalize=True if "jpe" not in self.config.n_model else False,
            )
            if "jpe" in self.config.n_model:
                return PororoJamoPairTokenizer(model, self.config)
            if "mecab.bpe" in self.config.n_model:
                return PororoMecabSPTokenizer(model, self.config)
            return PororoSPTokenizer(model, self.config)


class PororoTokenizerBase(PororoSimpleBase):

    @abstractmethod
    def detokenize(self, tokens: List[str]):
        raise NotImplementedError("`detokenize()` is not implemented")

    @abstractmethod
    def convert_tokens_to_ids(self, tokens: List[str]):
        raise NotImplementedError(
            "`convert_tokens_to_ids()` is not implemented")


class PororoSentTokenizer(PororoTokenizerBase):

    def __init__(self, model, config):
        super().__init__(config)
        self._model = model

    def cj_tokenize(self, text: str):
        text = text.replace("。", "。[SEP]")
        text = text.replace("！", "！[SEP]")
        text = text.replace("？", "？[SEP]")

        if "[SEP]" in text:
            sents = text.split("[SEP]")
            sents = sents[:-1]
        else:
            sents = [text]

        return sents

    def predict(self, text: str, **kwargs) -> List[str]:
        if self.lang in ["zh", "ja"]:
            return self.cj_tokenize(text)
        elif self.lang == "ko":
            return split_sentences(text)
        return self._model(text)


class PororoMecabKoTokenizer(PororoTokenizerBase):

    def __init__(self, model, config):
        super().__init__(config)
        self._model = model

    def detokenize(self, tokens: List[str]):
        text = "".join(tokens).replace("▃", " ").strip()
        return text

    def predict(
        self,
        text: str,
        **kwargs,
    ) -> List[str]:
        preserve_whitespace = kwargs.get("preserve_whitespace", True)

        text = text.strip()
        text_ptr = 0
        results = list()

        for unit in self._model.parse(text):
            token = unit[0]
            if preserve_whitespace:
                if text[text_ptr] == " ":
                    # Move text pointer to whitespace token to reserve whitespace
                    # cf. to prevent double white-space, we move pointer to next eojeol
                    while text[text_ptr] == " ":
                        text_ptr += 1
                    results.append(" ")
            results.append(token)
            text_ptr += len(token)

        return results


class PororoMosesTokenizer(PororoTokenizerBase):

    def __init__(self, model, detok, config):
        super().__init__(config)
        self._model = model
        self._detok = detok

    def detokenize(self, tokens: List[str]):
        return self._detok.detokenize(tokens)

    def predict(self, text: str, **kwargs) -> List[str]:
        return self._model.tokenize(text)


class PororoJiebaTokenizer(PororoTokenizerBase):

    def __init__(self, model, config):
        super().__init__(config)
        self._model = model

    def detokenize(self, tokens: List[str]):
        return "".join(tokens)

    def predict(self, text: str, **kwargs) -> List[str]:
        return list(self._model(text))


class PororoMecabTokenizer(PororoTokenizerBase):

    def __init__(self, model, config):
        super().__init__(config)
        self._model = model

    def detokenize(self, tokens: List[str]):
        return "".join(tokens)

    def predict(self, text: str, **kwargs) -> List[str]:
        parsed = self._model.parse(text)

        res = []
        for line in parsed.split("\n"):
            if line == "EOS":
                break
            toks = line.split("\t")
            res.append(toks[0])
        return res


class PororoWordTokenizer(PororoTokenizerBase):

    def __init__(self, config):
        super().__init__(config)

    def detokenize(self, tokens: List[str]) -> str:
        """
        Untokenizing a text undoes the tokenizing operation, restoring
        punctuation and spaces to the places that people expect them to be.
        Ideally, `untokenize(tokenize(text))` should be identical to `text`,
        except for line breaks.

        """
        text = " ".join(tokens)
        step1 = text.replace("`` ", '"').replace(" ''", '"')
        step1 = step1.replace(". . .", "...")

        step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")

        step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)

        step4 = re.sub(r" ([.,:;?!%]+)$", r"\1", step3)

        step5 = step4.replace(" '", "'").replace(" n't", "n't")
        step5 = step5.replace("can not", "cannot")

        step6 = step5.replace(" ` ", " '")
        return step6.strip()

    def predict(self, text: str, **kwargs) -> List[str]:
        return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)


class PororoCharTokenizer(PororoTokenizerBase):

    def __init__(self, config):
        super().__init__(config)

    def detokenize(self, tokens: List[str]):
        text = "".join(tokens).replace("▁", " ").strip()
        return text

    def predict(self, text: str, **kwargs) -> List[str]:
        text = text.strip().replace(" ", "▁")
        return list(text)


class PororoJamoTokenizer(PororoTokenizerBase):

    def __init__(self, config):
        super().__init__(config)

    def detokenize(self, tokens: List[str]):
        return normalize("NFKC", "".join(tokens)).replace("▁", " ")

    def predict(self, text: str, **kwargs) -> List[str]:
        return list("▁".join(
            [normalize("NFKD", token) for token in text.strip().split(" ")]))


class PororoJamoPairTokenizer(PororoTokenizerBase):

    def __init__(self, model, config):
        super().__init__(config)
        self._model = model

    def detokenize(self, tokens: List[str]):
        tokens = list("".join(tokens).replace("▁", " ").strip())
        return normalize("NFKC", "".join(tokens)).replace("▁", " ")

    def predict(self, text: str, **kwargs) -> List[str]:
        text = "▁".join(
            [normalize("NFKD", token) for token in text.strip().split(" ")])
        tokenized = self._model.segment(text.strip())
        return tokenized


class PororoSPTokenizer(PororoTokenizerBase):

    def __init__(self, model, config):
        super().__init__(config)
        self._model = model

    def detokenize(self, tokens: List[str]):
        text = "".join(tokens).replace("▁", " ").strip()
        return text

    def predict(self, text: str, **kwargs):
        tokenized = self._model.segment(text.strip())
        return tokenized


class PororoMecabSPTokenizer(PororoTokenizerBase):

    def __init__(self, model, config):
        super().__init__(config)
        self._model = model

    def detokenize(self, tokens: List[str]):
        text = "".join(tokens).replace("▁", " ").strip()
        return text

    def predict(self, text: str, **kwargs):
        tokenized = self._model.segment(text)
        return tokenized


class PororoRoBERTaTokenizer(PororoTokenizerBase):

    def __init__(self, model, vocab, inv_dict, config):
        super().__init__(config)
        self._model = model
        self._vocab = vocab
        self._inv_dict = inv_dict

    def convert_tokens_to_ids(self, tokens: List[str]):
        return [self._vocab[token] for token in tokens]

    def predict(self, text: str, **kwargs):
        tokens = self._model.encode(text)
        tokens = [self._inv_dict[token] for token in tokens]
        return tokens
