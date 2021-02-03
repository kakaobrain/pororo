"""Paraphrase Generation modeling class"""

from typing import Optional

from pororo.tasks.utils.base import PororoFactoryBase, PororoGenerationBase
from pororo.tasks.utils.download_utils import download_or_load


class PororoParaphraseFactory(PororoFactoryBase):
    """
    paraphrase generation using Transformer Seq2Seq

    Multi (`transformer.large.multi.mtpg`)

        - dataset: Internal data
        - metric: BLEU score

            +----------+------------+
            | Language | BLEU score |
            +==========+============+
            | Average  |   33.00    |
            +----------+------------+
            | Englosh  |   54       |
            +----------+------------+
            | Korean   |   50       |
            +----------+------------+
            | Japanese |   20       |
            +----------+------------+
            | Chinese  |   8        |
            +----------+------------+

    Multi (`transformer.large.multi.fast.mtpg`)

        - dataset: Internal data
        - metric: BLEU score

            +----------+------------+
            | Language | BLEU score |
            +==========+============+
            | Average  |   33.50    |
            +----------+------------+
            | Englosh  |   56       |
            +----------+------------+
            | Korean   |   50       |
            +----------+------------+
            | Japanese |   20       |
            +----------+------------+
            | Chinese  |   8        |
            +----------+------------+

    Args:
        text (str): input sentence to be paraphrase generated
        beam (int): beam search size
        temperature (float): temperature scale
        top_k (int): top-K sampling vocabulary size
        top_p (float): top-p sampling ratio
        no_repeat_ngram_size (int): no repeat ngram size
        len_penalty (float): length penalty ratio

    Returns:
        str: generated paraphrase

    Examples:
        >>> pg = Pororo(task="pg", lang="ko")
        >>> pg("노는게 제일 좋아. 친구들 모여라. 언제나 즐거워.")
        노는 것이 가장 좋습니다. 친구들끼리 모여 주세요. 언제나 즐거운 시간 되세요.
        >>> pg = Pororo("pg", lang="zh")
        >>> pg("我喜欢足球")  # 나는 축구를 좋아해
        '我喜欢球球球'  # 나는 공을 좋아해
        >>> pg = Pororo(task="pg", lang="ja")
        >>> pg("雨の日を聞く良い音楽をお勧めしてくれ。")  # 비오는 날 듣기 좋은 음악 가르쳐줘
        '雨の日を聞くいい音楽を教えてください。'          # 비오는 날 듣기 좋은 음악을 가르쳐 주세요
        >>> pg = Pororo("pg", lang="en")
        >>> pg("There is someone at the door.")
        "Someone's at the door."
        >>> pg("I'm good, but thanks for the offer.")
        "I'm fine, but thanks for the deal."

    """

    def __init__(self, task: str, lang: str, model: Optional[str]):
        super().__init__(task, lang, model)

    @staticmethod
    def get_available_langs():
        return ["en", "ko", "zh", "ja"]

    @staticmethod
    def get_available_models():
        return {
            "en": [
                "transformer.large.multi.mtpg",
                "transformer.large.multi.fast.mtpg",
                "transformer.base.en.pg",
            ],
            "ko": [
                "transformer.large.multi.mtpg",
                "transformer.large.multi.fast.mtpg",
                "transformer.base.ko.pg_long",
                "transformer.base.ko.pg",
            ],
            "zh": [
                "transformer.large.multi.mtpg",
                "transformer.large.multi.fast.mtpg",
                "transformer.base.zh.pg",
            ],
            "ja": [
                "transformer.large.multi.mtpg",
                "transformer.large.multi.fast.mtpg",
                "transformer.base.ja.pg",
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
        if "multi" in self.config.n_model:
            from fairseq.models.transformer import TransformerModel

            from pororo.tasks.utils.tokenizer import CustomTokenizer

            load_dict = download_or_load(
                f"transformer/{self.config.n_model}",
                "multi",
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

            return PororoTransformerTransMulti(
                model,
                self.config,
                tokenizer,
            )

        if "transformer" in self.config.n_model:
            from fairseq.models.transformer import TransformerModel

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

            if self.config.lang != "zh":
                from pororo.tasks.utils.tokenizer import CustomTokenizer

                tokenizer = CustomTokenizer.from_file(
                    vocab_filename=f"{load_dict.src_tok}/vocab.json",
                    merges_filename=f"{load_dict.src_tok}/merges.txt",
                )

            return PororoTransformerParaphrase(model, self.config, tokenizer)


class PororoTransformerTransMulti(PororoGenerationBase):

    def __init__(self, model, config, tokenizer):
        super().__init__(config)
        self._model = model
        self._tokenizer = tokenizer
        self._mapping = {"en": "_XX", "ja": "_XX", "ko": "_KR", "zh": "_CN"}

    def _langtok(self, lang: str):
        """
        Args:
            lang (str): language code

        See Also:
            https://github.com/pytorch/fairseq/blob/master/fairseq/data/multilingual/multilingual_utils.py#L34

        """
        return f"[{lang + self._mapping[lang]}]"

    def _preprocess(self, text: str) -> str:
        """
        Preprocess non-chinese input sentence to replace whitespace token with whitespace

        Args:
            text (str): non-chinese sentence

        Returns:
            str: preprocessed non-chinese sentence

        """
        if self.config.lang == "en":
            pieces = " ".join(self._tokenizer.segment(text.strip()))
        else:
            pieces = " ".join([c if c != " " else "▁" for c in text.strip()])
        return f"{self._langtok(self.config.lang)} {pieces} {self._langtok(self.config.lang)}"

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
            text (str): input sentence to be paraphrase generated
            beam (int): beam search size
            temperature (float): temperature scale
            top_k (int): top-K sampling vocabulary size
            top_p (float): top-p sampling ratio
            no_repeat_ngram_size (int): no repeat ngram size
            len_penalty (float): length penalty ratio

        Returns:
            str: machine translated sentence

        """
        text = self._preprocess(text)

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


class PororoTransformerParaphrase(PororoGenerationBase):

    def __init__(self, model, config, tokenizer):
        super().__init__(config)
        self._model = model
        self._tokenizer = tokenizer

    def _preprocess(self, text: str):
        """
        Preprocess non-chinese input sentence to replace whitespace token with whitespace

        Args:
            text (str): non-chinese sentence

        Returns:
            str: preprocessed non-chinese sentence

        """
        pieces = self._tokenizer.segment(text.strip())
        return " ".join(pieces)

    def _zh_preprocess(self, text: str):
        """
        Preprocess chinese input sentence to replace whitespace token with whitespace

        Args:
            text (str): chinese sentence

        Returns:
            str: preprocessed chinese sentence

        """
        return " ".join(char for char in text)

    def _postprocess(self, output: str):
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
        beam: int = 1,
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = -1,
        no_repeat_ngram_size: int = 4,
        len_penalty: float = 1.0,
        **kwargs,
    ):
        """
        Conduct paraphrase generation using Transformer Seq2Seq

        Args:
            text (str): input sentence to be paraphrase generated
            beam (int): beam search size
            temperature (float): temperature scale
            top_k (int): top-K sampling vocabulary size
            top_p (float): top-p sampling ratio
            no_repeat_ngram_size (int): no repeat ngram size
            len_penalty (float): length penalty ratio

        Returns:
            str: generated paraphrase

        """
        sampling = False

        if top_k != -1 or top_p != -1:
            sampling = True

        if self._tokenizer is not None:
            text = self._preprocess(text)
        else:
            text = self._zh_preprocess(text)

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
