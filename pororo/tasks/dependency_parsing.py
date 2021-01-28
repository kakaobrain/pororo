"""Dependency Parsing related modeling class"""

from typing import List, Optional, Tuple

from pororo.tasks.utils.base import PororoFactoryBase, PororoSimpleBase


class PororoDpFactory(PororoFactoryBase):
    """
    Conduct dependency parsing

    Korean (`posbert-base`)

        - dataset: https://corpus.korean.go.kr/ 구문 분석 말뭉치
        - metric: UAS (90.57), LAS (95.96)

    Args:
        sent: (str) sentence to be parsed dependency

    Returns:
        List[Tuple[int, str, int, str]]: token index, token label, token head and its relation

    Examples:
        >>> dp = Pororo(task="dep_parse", lang="ko")
        >>> dp("분위기도 좋고 음식도 맛있었어요. 한 시간 기다렸어요.")
        [(1, '분위기도', 2, 'NP_SBJ'), (2, '좋고', 4, 'VP'), (3, '음식도', 4, 'NP_SBJ'), (4, '맛있었어요.', 7, 'VP'), (5, '한', 6, 'DP'), (6, '시간', 7, 'NP_OBJ'), (7, '기다렸어요.', -1, 'VP')]
        >>> dp("한시간 기다렸어요.")
        [(1, '한시간', 2, 'NP_OBJ'), (2, '기다렸어요.', -1, 'VP')]

    """

    def __init__(self, task: str, lang: str, model: Optional[str]):
        super().__init__(task, lang, model)

    @staticmethod
    def get_available_langs():
        return ["ko"]

    @staticmethod
    def get_available_models():
        return {"ko": ["posbert.base.ko.dp", "charbert.base.ko.dp"]}

    def load(self, device: str):
        """
        Load user-selected task-specific model

        Args:
            device (str): device information

        Returns:
            object: User-selected task-specific model

        """
        from pororo.tasks import PororoPosFactory

        if "posbert" in self.config.n_model:
            from pororo.models.brainbert import RobertaSegmentModel

            model = (RobertaSegmentModel.load_model(
                f"bert/{self.config.n_model}",
                self.config.lang,
            ).eval().to(device))

            tagger = PororoPosFactory(
                task="pos",
                model="mecab-ko",
                lang=self.config.lang,
            ).load(device)

            return PororoSegmentBertDP(model, tagger, self.config)


class PororoSegmentBertDP(PororoSimpleBase):

    def __init__(self, model, tagger, config):
        super().__init__(config)
        self._tagger = tagger
        self._model = model

    def _preprocess(self, sent: str) -> Tuple:
        """
        Preprocess dependency parsing input

        Args:
            sent (str): input sentence to be preprocessed

        Returns:
            str: preprocessed input sentence with pos tag

        """
        pairs = self._tagger(sent, return_surface=True)
        tokens = ["<s>", "▃"
                 ] + [pair[0] if pair[0] != " " else "▃" for pair in pairs]
        tags = [
            pair[1] if pair[0] != " " else pairs[i + 1][1]
            for i, pair in enumerate(pairs)
        ]
        prefix = ["XX", tags[0]]
        tags = prefix + tags

        res_tags = []
        for tag in tags:
            if "+" in tag:
                tag = tag[:tag.find("+")]
            res_tags.append(tag)
        return tokens, res_tags

    def _postprocess(
        self,
        ori: str,
        tokens: List[str],
        heads: List[int],
        labels: List[str],
    ):
        """
        Postprocess dependency parsing output

        Args:
            ori (sent): original sentence
            heads (List[str]): dependency heads generated by model
            labels (List[str]): tag labels generated by model

        Returns:
            List[Tuple[int, str, int, str]]: token index, token label, token head and its relation

        """
        eojeols = ori.split()

        indices = [i for i, token in enumerate(tokens) if token == "▃"]
        real_heads = [head for i, head in enumerate(heads) if i in indices]
        real_labels = [label for i, label in enumerate(labels) if i in indices]

        result = []
        for i, (head, label,
                eojeol) in enumerate(zip(
                    real_heads,
                    real_labels,
                    eojeols,
                )):
            curr = i + 1

            try:
                head_eojeol = indices.index(head) + 1
            except:
                head_eojeol = -1

            if head_eojeol == curr:
                head_eojeol = -1

            result.append((curr, eojeol, head_eojeol, label))
        return result

    def predict(self, sent: str):
        """
        Conduct dependency parsing

        Args:
            sent: (str) sentence to be parsed dependency

        Returns:
            List[Tuple[int, str, int, str]]: token index, token label, token head and its relation

        """
        tokens, tags = self._preprocess(sent)
        heads, labels = self._model.predict_dependency(tokens, tags)
        heads = [int(head) - 1 for head in heads]  # due to default <s> token
        return self._postprocess(sent, tokens, heads, labels)
