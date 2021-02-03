"""Machine-translation related modeling class"""

from typing import Optional

from pororo.tasks.utils.base import PororoFactoryBase, PororoGenerationBase
from pororo.tasks.utils.download_utils import download_or_load


class PororoTranslationFactory(PororoFactoryBase):
    """
    Machine translation using Transformer models

    Multi (`transformer.large.multi.mtpg`)

        - dataset: Train (Internal data) / Test (Multilingual TED Talk)
        - metric: BLEU score

            +-----------------+-----------------+------------+
            | Source Language | Target Language | BLEU score |
            +=================+=================+============+
            | Average         |  X              |   10.00    |
            +-----------------+-----------------+------------+
            | English         |  Korean         |   15       |
            +-----------------+-----------------+------------+
            | English         |  Japanese       |   8        |
            +-----------------+-----------------+------------+
            | English         |  Chinese        |   8        |
            +-----------------+-----------------+------------+
            | Korean          |  English        |   15       |
            +-----------------+-----------------+------------+
            | Korean          |  Japanese       |   10       |
            +-----------------+-----------------+------------+
            | Korean          |  Chinese        |   4        |
            +-----------------+-----------------+------------+
            | Japanese        |  English        |   11       |
            +-----------------+-----------------+------------+
            | Japanese        |  Korean         |   13       |
            +-----------------+-----------------+------------+
            | Japanese        |  Chinese        |   4        |
            +-----------------+-----------------+------------+
            | Chinese         |  English        |   16       |
            +-----------------+-----------------+------------+
            | Chinese         |  Korean         |   10       |
            +-----------------+-----------------+------------+
            | Chinese         |  Japanese       |   6        |
            +-----------------+-----------------+------------+

        - ref: http://www.cs.jhu.edu/~kevinduh/a/multitarget-tedtalks/
        - note: This result is about out of domain settings, TED Talk data wasn't used during model training.

    Multi (`transformer.large.multi.fast.mtpg`)

        - dataset: Train (Internal data) / Test (Multilingual TED Talk)
        - metric: BLEU score

            +-----------------+-----------------+------------+
            | Source Language | Target Language | BLEU score |
            +=================+=================+============+
            | Average         |  X              |   8.75     |
            +-----------------+-----------------+------------+
            | English         |  Korean         |   13       |
            +-----------------+-----------------+------------+
            | English         |  Japanese       |   6        |
            +-----------------+-----------------+------------+
            | English         |  Chinese        |   7        |
            +-----------------+-----------------+------------+
            | Korean          |  English        |   15       |
            +-----------------+-----------------+------------+
            | Korean          |  Japanese       |   11       |
            +-----------------+-----------------+------------+
            | Korean          |  Chinese        |   10       |
            +-----------------+-----------------+------------+
            | Japanese        |  English        |   3        |
            +-----------------+-----------------+------------+
            | Japanese        |  Korean         |   13       |
            +-----------------+-----------------+------------+
            | Japanese        |  Chinese        |   4        |
            +-----------------+-----------------+------------+
            | Chinese         |  English        |   15       |
            +-----------------+-----------------+------------+
            | Chinese         |  Korean         |   8        |
            +-----------------+-----------------+------------+
            | Chinese         |  Japanese       |   4        |
            +-----------------+-----------------+------------+

        - ref: http://www.cs.jhu.edu/~kevinduh/a/multitarget-tedtalks/
        - note: This result is about out of domain settings, TED Talk data wasn't used during model training.

    Args:
        text (str): input text to be translated
        beam (int): beam search size
        temperature (float): temperature scale
        top_k (int): top-K sampling vocabulary size
        top_p (float): top-p sampling ratio
        no_repeat_ngram_size (int): no repeat ngram size
        len_penalty (float): length penalty ratio

    Returns:
        str: machine translated sentence

    Examples:
        >>> mt = Pororo(task="translation", lang="multi")
        >>> mt("케빈은 아직도 일을 하고 있다.", src="ko", tgt="en")
        'Kevin is still working.'
        >>> mt("死神は りんごしか食べない。", src="ja", tgt="ko")
        '사신은 사과밖에 먹지 않는다.'
        >>> mt("人生的伟大目标，不是知识而是行动。", src="zh", tgt="ko")
        '인생의 위대한 목표는 지식이 아니라 행동이다.'

    """

    def __init__(
        self,
        task: str,
        lang: str,
        model: Optional[str],
        tgt: str = None,
    ):
        super().__init__(task, lang, model)
        self._src = self.config.lang
        self._tgt = tgt

    @staticmethod
    def get_available_langs():
        return ["multi"]

    @staticmethod
    def get_available_models():
        return {
            "multi": [
                "transformer.large.multi.mtpg",
                "transformer.large.multi.fast.mtpg",
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
        from pororo.tasks import PororoTokenizationFactory

        sent_tokenizer = (lambda text, lang: PororoTokenizationFactory(
            task="tokenization",
            lang=lang,
            model=f"sent_{lang}",
        ).load(device).predict(text))

        if "multi" in self.config.n_model:
            from fairseq.models.transformer import TransformerModel

            from pororo.tasks.utils.tokenizer import CustomTokenizer

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

            tokenizer = CustomTokenizer.from_file(
                vocab_filename=f"{load_dict.src_tok}/vocab.json",
                merges_filename=f"{load_dict.src_tok}/merges.txt",
            )

            if "mtpg" in self.config.n_model:
                langtok_style = "mbart"
            elif "m2m" in self.config.n_model:
                langtok_style = "multilingual"
            else:
                langtok_style = "basic"

            return PororoTransformerTransMulti(
                model,
                self.config,
                tokenizer,
                sent_tokenizer,
                langtok_style,
            )


class PororoTransformerTransMulti(PororoGenerationBase):

    def __init__(self, model, config, tokenizer, sent_tokenizer, langtok_style):
        super().__init__(config)
        self._model = model
        self._tokenizer = tokenizer
        self._sent_tokenizer = sent_tokenizer
        self._langtok_style = langtok_style

    def _langtok(self, lang: str, langtok_style: str):
        """
        Args:
            lang (str): language
            langtok_style (str): style of language token

        See Also:
            https://github.com/pytorch/fairseq/blob/master/fairseq/data/multilingual/multilingual_utils.py#L34

        """
        if langtok_style == "basic":
            return f"[{lang.upper()}]"

        elif langtok_style == "mbart":
            mapping = {"en": "_XX", "ja": "_XX", "ko": "_KR", "zh": "_CN"}
            return f"[{lang + mapping[lang]}]"

        elif langtok_style == "multilingual":
            return f"__{lang}__"

    def _preprocess(self, text: str, src: str, tgt: str) -> str:
        """
        Preprocess non-chinese input sentence to replace whitespace token with whitespace

        Args:
            text (str): non-chinese sentence
            src (str): source language
            tgt (str): target language

        Returns:
            str: preprocessed non-chinese sentence

        """
        if src == "en":
            pieces = " ".join(self._tokenizer.segment(text.strip()))
        else:
            pieces = " ".join([c if c != " " else "▁" for c in text.strip()])
        return f"{self._langtok(src, self._langtok_style)} {pieces} {self._langtok(tgt, self._langtok_style)}"

    def _postprocess(self, output: str) -> str:
        """
        Postprocess output sentence to replace whitespace

        Args:
            output (str): output sentence generated by model

        Returns:
            str: postprocessed output sentence

        """
        return output.replace(" ", "").replace("▁", " ").strip()

    def predict(
        self,
        text: str,
        src: str,
        tgt: str,
        beam: int = 5,
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = -1,
        no_repeat_ngram_size: int = 4,
        len_penalty: float = 1.0,
        **kwargs,
    ) -> str:
        """
        Conduct machine translation

        Args:
            text (str): input text to be translated
            beam (int): beam search size
            temperature (float): temperature scale
            top_k (int): top-K sampling vocabulary size
            top_p (float): top-p sampling ratio
            no_repeat_ngram_size (int): no repeat ngram size
            len_penalty (float): length penalty ratio

        Returns:
            str: machine translated sentence

        """
        text = self._preprocess(text, src, tgt)

        sampling = False

        if top_k != -1 or top_p != -1:
            sampling = True

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
        return output

    def __call__(
        self,
        text: str,
        src: str,
        tgt: str,
        beam: int = 5,
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = -1,
        no_repeat_ngram_size: int = 4,
        len_penalty: float = 1.0,
    ):
        assert isinstance(text, str), "Input text should be string type"

        assert src in [
            "ko",
            "zh",
            "ja",
            "en",
        ], "Source language must be one of CJKE !"

        assert tgt in [
            "ko",
            "zh",
            "ja",
            "en",
        ], "Target language must be one of CJKE !"

        return " ".join([
            self.predict(
                t,
                src,
                tgt,
                beam,
                temperature,
                top_k,
                top_p,
                no_repeat_ngram_size,
                len_penalty,
            ) for t in self._sent_tokenizer(text, src)
        ])
