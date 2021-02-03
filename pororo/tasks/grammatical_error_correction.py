"""Grammatical Error Correction related modeling class"""

import re
from typing import List, Optional, Union

from pororo.tasks.utils.base import (
    PororoFactoryBase,
    PororoGenerationBase,
    PororoSimpleBase,
)
from pororo.tasks.utils.download_utils import download_or_load


class PororoGecFactory(PororoFactoryBase):
    """
    Grammatical error correction

    English (`transformer.base.en.gec`)

        - dataset: FCE, W&I+LOCNESS
        - metric: TBU

    English (`transformer.base.en.char_gec`)

        - dataset: xfspell
        - metric: TBU
        - ref: http://www.realworldnlpbook.com/blog/unreasonable-effectiveness-of-transformer-spell-checker.html

    Korean (`charbert.base.ko.spacing`)

        - dataset: Internal data (based on Wikipedia)
        - metric: F1 (89.51)

    Args:
        text (str): input sentence to fix grammatical error
        beam (int): size of beam search
        temperature (float): temperature for sampling
        top_k (int): variable for top k sampling
        top_p (float): variable for top p sampling
        no_repeat_ngram_size (int): no repeat ngram size
        len_penalty (float): length penalty ratio

    Examples:
        >>> gec = Pororo(task="gec", lang="en")
        >>> gec("This apple are so sweet.")
        "This apple is so sweet."
        >>> gec("'I've love you, before I meet her!'")
        "'I've loved you, before I met her!"
        >>> # It works better if I use two modules in succession with `correct_spell` option
        >>> # Of course, it requires more computation and time.
        >>> gec("Travel by bus is exspensive , bored and annoying .") # bad result
        'Travel by bus is exspensive, boring and annoying.'
        >>> gec("Travel by bus is exspensive , bored and annoying .", correct_spell=True) # better result
        'Travelling by bus is expensive, boring, and annoying.'
        >>> spacing = Pororo(task="gec", lang="ko")
        >>> spacing("카 카오브 레인에서는 무슨 일을 하 나 요?")
        '카카오브레인에서는 무슨 일을 하나요?'
        >>> spacing("아버지가방에들어간다.")
        '아버지가 방에 들어간다.'


    Notes:
        Korean error correction is beta version.
        It only supports spacing correction currently.

    """

    def __init__(self, task: str, lang: str, model: Optional[str]):
        super().__init__(task, lang, model)

    @staticmethod
    def get_available_langs():
        return ["en", "ko"]

    @staticmethod
    def get_available_models():
        return {
            "en": [
                "transformer.base.en.gec",
                "transformer.base.en.char_gec",
            ],
            "ko": ["charbert.base.ko.spacing"],
        }

    def load(self, device: str):
        """
        Load user-selected task-specific model

        Args:
            device (str): device information

        Returns:
            object: User-selected task-specific model

        """

        if "charbert" in self.config.n_model:
            from pororo.models.brainbert import CharBrainRobertaModel

            model = (CharBrainRobertaModel.load_model(
                f"bert/{self.config.n_model}",
                self.config.lang,
            ).eval().to(device))
            print(
                "As of now, this beta model tries to correct spacing errors in Korean text."
            )
            return PororoBertSpacing(model, self.config)

        if "transformer" in self.config.n_model:
            from fairseq.models.transformer import TransformerModel

            from pororo.tasks.utils.tokenizer import CustomTokenizer

            load_dict = download_or_load(
                f"transformer/{self.config.n_model}",
                self.config.lang,
            )

            tokenizer = None
            model = (TransformerModel.from_pretrained(
                model_name_or_path=load_dict.path,
                checkpoint_file=f"{self.config.n_model}.pt",
                data_name_or_path=load_dict.dict_path,
                source_lang=load_dict.src_dict,
                target_lang=load_dict.tgt_dict,
            ).eval().to(device))

            if "char" in self.config.n_model:
                return PororoTransformerGecChar(model, self.config)

            if load_dict.src_tok:
                tokenizer = CustomTokenizer.from_file(
                    vocab_filename=f"{load_dict.src_tok}/vocab.json",
                    merges_filename=f"{load_dict.src_tok}/merges.txt",
                )

            return PororoTransformerGec(model, tokenizer, device, self.config)


class PororoTransformerGecChar(PororoGenerationBase):

    def __init__(self, model, config):
        super().__init__(config)
        self._model = model
        self._symbols = "[:.,!?\"']"
        self._chars = set(
            "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?'-"
        )

    def _preprocess(self, text: str):
        """
        Preprocess input sentence to replace whitespace token with whitespace"

        Args:
            text (str): input sentence

        Returns:
            str: preprocessed input sentence

        """
        text = text.strip()
        chars = []
        unks = []
        for ch in text:
            if ch in self._chars:
                chars.append(ch)
            else:
                chars.append("▮")
                unks.append(ch)
        text = "".join(chars)
        text = re.sub(" +", " ", text).strip()
        tokens = [ch if ch != " " else "▁" for ch in text]
        return " ".join(tokens[:1023]), unks

    def _despace_puncts(self, text: str):
        """
        desapce punctionation

        Args:
            text (str): input sentence

        Returns:
            desapced sentence

        """
        text = re.sub("([\"']) ", r"\1", text, count=0)
        return re.sub(f" ({self._symbols}+)", r"\1", text, count=0)

    def _postprocess(self, output: str, unks: List):
        """
        Postprocess output sentence to replace whitespace

        Args:
            output (str): output sentence generated by model
            unks (List[str]): pre-replaced unknown token lists

        Returns:
            str: postprocessed output sentence

        """
        if unks:
            pointer = 0
            chars = [c for c in output]
            for i, char in enumerate(chars):
                if char == "▮":
                    chars[i] = unks[pointer]
                    pointer += 1
            output = "".join(chars)
        output = output.replace(" ", "").replace("▁", " ").strip()
        return output

    def predict(
        self,
        text: str,
        beam: int = 5,
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = -1,
        no_repeat_ngram_size: int = 4,
        len_penalty: float = 1.0,
    ):
        """
        Conduct grammar error correction

        Args:
            text (str): input sentence
            beam (int): beam search size
            temperature (float): temperature scale
            top_k (int): top-K sampling vocabulary size
            top_p (float): top-p sampling ratio
            no_repeat_ngram_size (int): no repeat ngram size
            len_penalty (float): length penalty ratio

        Returns:
            str: grammartically corrected sentence

        """
        sampling = False

        if top_k != -1 or top_p != -1:
            sampling = True

        text, unks = self._preprocess(text)

        output = self._model.translate(
            text,
            beam=beam,
            sampling=sampling,
            temperature=temperature,
            sampling_topk=top_k,
            sampling_topp=top_p,
            max_len_a=1,
            max_len_b=50,
            no_repeat_ngram_size=no_repeat_ngram_size,
            lenpen=len_penalty,
        )
        output = self._postprocess(output, unks)

        return output


class PororoTransformerGec(PororoGenerationBase):

    def __init__(self, model, tokenizer, device, config):
        super().__init__(config)
        self._model = model
        self._tokenizer = tokenizer

        self._symbols = "[:.,!?\"']"
        self._clitics = ("n't", "'ll", "'s", "'m", "'ve", "'d", "'re")

        self._device = device
        self._corrector = None

    def _space_puncts(self, text: str):
        """
        Args:
            text (str): input sentence

        Returns:
            str: processed string

        Examples:
            noise!He -> noise ! He

        """
        _text = []
        for word in text.strip().split():
            detect_clitic = False
            for clitic in self._clitics:
                if re.search(clitic, word):
                    detect_clitic = True
            if re.search(self._symbols, word) is not None and not detect_clitic:
                if not word.count(".") > 1:  # e.g., `U.S.` is correct.
                    word = re.sub(f"({self._symbols})", r" \1 ", word, count=0)
            _text.append(word)

        return " ".join(_text)

    def _space_contracts(self, text: str):
        """
        Args:
            text (str): input sentence

        Returns:
            str: processed string

        Examples:
            haven't -> have n't
        """
        _text = []
        for w in text.split():
            for clitic in self._clitics:
                w = re.sub(f"([A-Za-z])({clitic})", r"\1 \2", w)
            _text.append(w)
        return " ".join(_text)

    def _collapse_spaces(self, text):
        """
        Args:
            text (str): input string

        Returns:
            str: processed string

        """
        text = re.sub(" +", " ", text)
        return text

    def _despace_puncts(self, text: str):
        """
        Inverse function of _space_puncts(self, text)

        Args:
            text (str): input sentence

        Returns:
            str: processed string

        """
        text = re.sub("([\"']) ", r"\1", text, count=0)
        return re.sub(f" ({self._symbols}+)", r"\1", text, count=0)

    def _despace_contracts(self, text: str):
        """
        Inverse function of _space_contracts(self, text)

        Args:
            text (str): input sentence

        Returns:
            str: processed string

        """
        for clitic in self._clitics:
            text = re.sub(f" ({clitic})", r"\1", text, count=0)
        return text

    def _preprocess(self, text: str):
        """
        Preprocess using simple methods

        Args:
            text (str): input sentence

        Returns:
            str: preprocessed string

        """
        text = self._space_puncts(text)
        text = self._space_contracts(text)
        text = self._collapse_spaces(text)
        pieces = self._tokenizer.segment(text.strip())
        return " ".join(pieces)

    def _postprocess(self, output: str):
        """
        Postprocess output sentence to replace whitespace

        Args:
            output (str): sentence to postprocess

        Returns:
            str: postprocessed string

        """
        output = output.replace(" ", "").replace("▁", " ").strip()
        output = self._despace_puncts(output)
        output = self._despace_contracts(output)
        return output

    def _correct_spell(self, text: str):
        """
        Conduct error correction for spell

        Args:
            text (str): input sentence

        Returns:
            result of spell error correction

        """
        if self._corrector is None:
            self._corrector = PororoGecFactory(
                task="gec",
                lang="en",
                model="transformer.base.en.char_gec",
            )
            self._corrector = self._corrector.load(self._device)
        return self._grammar_postprocess(self._corrector(text))

    def _grammar_postprocess(self, text: str):
        """
        Postprocess output sentence

        Args:
            text (str): sentence to postprocess

        Returns:
            str: postprocessed string

        """
        text = re.sub(" '(s|t|ll|m|re|ve|d)", r"'\1", text)
        # pairs
        for pair in ("()", "<>", "{}", "[]", "''", '""'):
            opening, closing = pair
            opening_ = re.escape(opening)
            closing_ = re.escape(closing)
            text = re.sub(
                f"{opening_} +([^{closing_}]+) +{closing_}",
                rf"{opening}\1{closing}",
                text,
            )

        text = re.sub(" ([:.,!?])", r"\1", text)
        return text

    def predict(
        self,
        text: str,
        beam: int = 5,
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = -1,
        no_repeat_ngram_size: int = 4,
        len_penalty: float = 1.0,
        **kwargs,
    ):
        """
        Conduct grammar error correction

        Args:
            text (str): input sentence
            beam (int): beam search size
            temperature (float): temperature scale
            top_k (int): top-K sampling vocabulary size
            top_p (float): top-p sampling ratio
            no_repeat_ngram_size (int): no repeat ngram size
            len_penalty (float): length penalty ratio

        Returns:
            str: grammartically corrected sentence

        Examples:
            >>> gec = Pororo(task="gec", model="transformer.base.en.gec", lang="en")
            >>> gec("This apple are so sweet.")
            "This apple is so sweet."
            >>> gec("'I've love you, before I meet her!'")
            "'I've loved you, before I met her!"

        """
        correct_spell = kwargs.get("correct_spell", False)

        sampling = False

        if top_k != -1 or top_p != -1:
            sampling = True

        if correct_spell:
            text = self._correct_spell(text)

        text = self._preprocess(text)
        output = self._model.translate(
            text,
            beam=beam,
            sampling=sampling,
            temperature=temperature,
            sampling_topk=top_k,
            sampling_topp=top_p,
            max_len_a=1,
            max_len_b=50,
            no_repeat_ngram_size=no_repeat_ngram_size,
            lenpen=len_penalty,
        )
        output = self._postprocess(output)
        return self._grammar_postprocess(output) if correct_spell else output


class PororoBertSpacing(PororoSimpleBase):

    def __init__(self, model, config):
        super().__init__(config)
        self._model = model

    def _postprocess(self, tokens: List[str]) -> str:
        """
        Postprocess spacing correction result

        Args:
            tokens (List[str]): list containing character and its predicted label

        Returns:
            str: postprocessed and spacing corrected sentence

        """
        result = str()

        for pair in tokens:
            token, label = pair

            if label == "0":
                result += token
            elif label == "1":
                result += f"▁{token}"
            elif label == "2":
                result += f"{token.replace('▁', '')}"

        return result.replace("▁", " ").strip()

    def predict(self, text: str, **kwargs) -> Union[List[str], str]:
        """
        Conduct spacing correction

        Args:
            text: (str) sentence to be spacing error corrected

        Returns:
            str: spacing error corrected sentence

        """
        if isinstance(text, str):
            text = [text]

        result = self._model.predict_tags(text)

        li_result = []
        for r in result:
            li_result.append(self._postprocess(r))

        return li_result if len(li_result) > 1 else li_result[0]
