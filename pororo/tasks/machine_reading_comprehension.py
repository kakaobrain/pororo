"""Reading Comprehension related modeling class"""

from typing import Optional, Tuple

from pororo.tasks.utils.base import PororoBiencoderBase, PororoFactoryBase


class PororoMrcFactory(PororoFactoryBase):
    """
    Conduct machine reading comprehesion with query and its corresponding context

    Korean (`brainbert.base.ko.korquad`)

        - dataset: KorQuAD 1.0 (Lim et al. 2019)
        - metric: EM (84.33), F1 (93.31)

    Args:
        query: (str) query string used as query
        context: (str) context string used as context

    Returns:
        Tuple[str, Tuple[int, int]]: predicted answer span and its indices

    Examples:
        >>> mrc = Pororo(task="mrc", lang="ko")
        >>> mrc(
        >>>    "카카오브레인이 공개한 것은?",
        >>>    "카카오 인공지능(AI) 연구개발 자회사 카카오브레인이 AI 솔루션을 첫 상품화했다. 카카오는 카카오브레인 '포즈(pose·자세분석) API'를 유료 공개한다고 24일 밝혔다. 카카오브레인이 AI 기술을 유료 API를 공개하는 것은 처음이다. 공개하자마자 외부 문의가 쇄도한다. 포즈는 AI 비전(VISION, 영상·화면분석) 분야 중 하나다. 카카오브레인 포즈 API는 이미지나 영상을 분석해 사람 자세를 추출하는 기능을 제공한다."
        >>> )
        ('포즈(pose·자세분석) API', (33, 44))
        >>> # when mecab doesn't work well for postprocess, you can set `postprocess` option as `False`
        >>> mrc("카카오브레인이 공개한 라이브러리 이름은?", "카카오브레인은 자연어 처리와 음성 관련 태스크를 쉽게 수행할 수 있도록 도와 주는 라이브러리 pororo를 공개하였습니다.", postprocess=False)
        ('pororo', (30, 34))

    """

    def __init__(self, task: str, lang: str, model: Optional[str]):
        super().__init__(task, lang, model)

    @staticmethod
    def get_available_langs():
        return ["ko"]

    @staticmethod
    def get_available_models():
        return {"ko": ["brainbert.base.ko.korquad"]}

    def load(self, device: str):
        """
        Load user-selected task-specific model

        Args:
            device (str): device information

        Returns:
            object: User-selected task-specific model

        """
        if "brainbert" in self.config.n_model:
            try:
                import mecab
            except ModuleNotFoundError as error:
                raise error.__class__(
                    "Please install python-mecab-ko with: `pip install python-mecab-ko`"
                )
            from pororo.models.brainbert import BrainRobertaModel
            from pororo.utils import postprocess_span

            model = (BrainRobertaModel.load_model(
                f"bert/{self.config.n_model}",
                self.config.lang,
            ).eval().to(device))

            tagger = mecab.MeCab()

            return PororoBertMrc(model, tagger, postprocess_span, self.config)


class PororoBertMrc(PororoBiencoderBase):

    def __init__(self, model, tagger, callback, config):
        super().__init__(config)
        self._model = model
        self._tagger = tagger
        self._callback = callback

    def predict(
        self,
        query: str,
        context: str,
        **kwargs,
    ) -> Tuple[str, Tuple[int, int]]:
        """
        Conduct machine reading comprehesion with query and its corresponding context

        Args:
            query: (str) query string used as query
            context: (str) context string used as context
            postprocess: (bool) whether to apply mecab based postprocess

        Returns:
            Tuple[str, Tuple[int, int]]: predicted answer span and its indices

        """
        postprocess = kwargs.get("postprocess", True)

        pair_result = self._model.predict_span(query, context)
        span = self._callback(
            self._tagger,
            pair_result[0],
        ) if postprocess else pair_result[0]

        return (
            span,
            pair_result[1],
        )
