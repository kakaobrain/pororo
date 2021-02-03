"""Fill-in-the-blank related modeling class"""

from typing import List, Optional

from pororo.tasks.utils.base import PororoFactoryBase, PororoSimpleBase


class PororoBlankFactory(PororoFactoryBase):
    """
    Conduct fill-in-the-blank with one __ token

    English (`roberta.base.en`)

        - dataset: N/A
        - metric: N/A

    Korean (`posbert.base.ko`)

        - dataset: N/A
        - metric: N/A

    Japanese (`jaberta.base.ja`)

        - dataset: N/A
        - metric: N/A

    Chinese (`zhberta.base.zh`)

        - dataset: N/A
        - metric: N/A

    Args:
        sent(str): input sentence which contains one __ token

    Returns:
        List[str]: token candidates could be fitted into __ token

    Examples:
        >>> fib = Pororo(task="fib", lang="en")
        >>> fib("David Beckham is a famous __ player.")
        ['football', 'soccer', 'basketball', 'baseball', 'sports']
        >>> fib = Pororo(task="fib", lang="ko")
        >>> fib("손흥민은 __의 축구선수이다.")
        ['대한민국', '잉글랜드', '독일', '스웨덴', '네덜란드', '덴마크', '미국', '웨일스', '노르웨이', '벨기에', '프랑스', '국적', '일본', '한국']
        >>> fib = Pororo(task="fib", lang="ja")
        >>> fib("日本の首都は__である。")
        ['東京', '大阪', '仙台', '釧路', '北海道']
        >>> fib = Pororo(task="fib", lang="zh")
        >>> fib("三__男子在街上做同样的舞蹈。")
        ['个', '名', '位', '女', '组']

    """

    def __init__(self, task: str, lang: str, model: Optional[str]):
        super().__init__(task, lang, model)

    @staticmethod
    def get_available_langs():
        return ["en", "ko", "ja", "zh"]

    @staticmethod
    def get_available_models():
        return {
            "en": ["roberta.base.en"],
            "ko": ["posbert.base.ko"],
            "ja": ["jaberta.base.ja"],
            "zh": ["zhberta.base.zh"],
        }

    def load(self, device: str):
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
            return PororoBertBlank(model, self.config)

        if "posbert" in self.config.n_model:
            try:
                import mecab  # noqa
            except ModuleNotFoundError as error:
                raise error.__class__(
                    "Please install python-mecab-ko with: `pip install python-mecab-ko`"
                )
            from pororo.models.brainbert import PosRobertaModel

            model = (PosRobertaModel.load_model(
                f"bert/{self.config.n_model}",
                self.config.lang,
            ).eval().to(device))
            return PororoBertBlank(model, self.config)

        if "jaberta" in self.config.n_model:
            from pororo.models.brainbert import JabertaModel

            model = (JabertaModel.load_model(
                f"bert/{self.config.n_model}",
                self.config.lang,
            ).eval().to(device))
            return PororoBertBlank(model, self.config)

        if "zhberta" in self.config.n_model:
            from pororo.models.brainbert import ZhbertaModel

            model = (ZhbertaModel.load_model(
                f"bert/{self.config.n_model}",
                self.config.lang,
            ).eval().to(device))
            return PororoBertBlank(model, self.config)


class PororoBertBlank(PororoSimpleBase):

    def __init__(self, model, config):
        super().__init__(config)
        self._model = model
        self._specials = [
            "<unk>",
            "<pad>",
            "<s>",
            "</s>",
            "<BOS>",
            "<EOS>",
            "▃",
            ",",
            ".",
            "?",
            "!",
            "/",
            "'",
            '"',
        ]

    def predict(self, sent: str, **kwargs) -> List[str]:
        """
        Conduct fill-in-the-blank with one __ token

        Args:
            sent(str): input sentence which contains one __ token

        Returns:
            List[str]: token candidates could be fitted into __ token

        """

        return [
            token.strip()
            for token in self._model.fill_mask(sent)
            if token not in self._specials
        ]
