"""Sentence Embedding related modeling class"""

from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import util

from pororo.tasks.utils.base import PororoFactoryBase, PororoSimpleBase
from pororo.tasks.utils.download_utils import download_or_load


class PororoSentenceFactory(PororoFactoryBase):
    """
    Sentence embedding based embedding vector

    English (`stsb-roberta-base`, `stsb-roberta-large`, `stsb-bert-base`, `stsb-bert-large`, `stsb-distllbert-base`)

        - dataset: N/A
        - metric : N/A

    Korean (`brainsbert.base.ko.kornli.korsts`)

        - dataset: N/A
        - metric: N/A

    Japanese (`jasbert.base.ja.nli.sts`)

        - dataset: N/A
        - metric: N/A

    Chinese (`zhsbert.base.zh.nli.sts`)

        - dataset: N/A
        - metric: N/A

    Examples:
        >>> se = Pororo(task="sentence_embedding", lang="ko")
        >>> se("나는 동물을 좋아하는 사람이야")
        [128.78, 200.12, 245.321, ...]  # (1, hidden dim)

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
                "stsb-roberta-base",
                "stsb-roberta-large",
                "stsb-bert-base",
                "stsb-bert-large",
                "stsb-distillbert-base",
            ],
            "ko": ["brainsbert.base.ko.kornli.korsts"],
            "ja": ["jasbert.base.ja.nli.sts"],
            "zh": ["zhsbert.base.zh.nli.sts"],
        }

    def load(self, device: str):
        """
        Load user-selected task-specific model

        Args:
            device (str): device information

        Returns:
            object: User-selected task-specific model

        """
        from sentence_transformers import SentenceTransformer

        model_path = self.config.n_model

        if self.config.lang != "en":
            model_path = download_or_load(
                f"sbert/{self.config.n_model}",
                self.config.lang,
            )
        model = SentenceTransformer(model_path).eval().to(device)
        return PororoSBertSentence(model, self.config)


class PororoSBertSentence(PororoSimpleBase):

    def __init__(self, model, config):
        super().__init__(config)
        self._model = model

    def find_similar_sentences(
        self,
        query: str,
        cands: List[str],
    ) -> Dict:
        """
        Conduct find similar sentences

        Args:
            query (str): query sentence to be acted as anchor
            cands (List[str]): candidate sentences to be compared

        Returns:
            Dict[str, List[Tuple[str, float]]]: list of tuple containing candidate sentence and its score

        Examples:
            >>> se = Pororo(task="sentence_embedding")
            >>> query = "He is the tallest person in the world"
            >>> cands = [
            >>>     "I hate this guy.",
            >>>     "You are so lovely!.",
            >>>     "Tom is taller than Jim."
            >>> ]
            >>> se.find_similar_sentences(query, cands)
            {
                'query': 'He is the tallest person in the world',
                'ranking': [(2, 'Tom is taller than Jim.', 0.49), (1, 'You are so lovely!.', 0.47), (0, 'I hate this guy.', 0.22)]
            }
            >>> se = Pororo(task="sentence_embedding", lang="ko")
            >>> query = "고양이가 창 밖을 바라본다"
            >>> cands = [
            >>>    "고양이가 카메라를 켠다",
            >>>    "남자와 여자가 걷고 있다",
            >>>    "고양이가 개를 만지려 하고 있다",
            >>>    "두 마리의 고양이가 창문을 보고 있다",
            >>>    "테이블 위에 앉아 있는 고양이가 창밖을 내다보고 있다",
            >>>    "창밖을 내다보는 고양이"
            >>> ]
            >>> se.find_similar_sentences(query, cands)
            {
                'query': '고양이가 창 밖을 바라본다',
                 'ranking': [(5, '창밖을 내다보는 고양이', 0.93), (4, '테이블 위에 앉아 있는 고양이가 창밖을 내다보고 있다', 0.91), (3, '두 마리의 고양이가 창문을 보고 있다', 0.78), (0, '고양이가 카메라를 켠다', 0.74), (2, '고양이가 개를 만지려 하고 있다', 0.41)]
            }
            >>> se = Pororo(task="sentence_embedding", lang="ja")
            >>> query = "おはようございます"  # Good morning
            >>> cands = ["こんにちは", "失礼します", "こんばんは"]  # Hello | Please Excuse Me (for Leaving) | Good evening
            >>> se.find_similar_sentences(query, cands)
            {
                'query': 'おはようございます',
                'ranking': [(0, 'こんにちは', 0.58), (2, 'こんばんは', 0.48), (1, '失礼します', 0.27)]
            }
            >>> se = Pororo(task="sentence_embedding", lang="zh")
            >>> query = "欢迎光临"  # Welcome
            >>> cands = ["你好。", "你会说英语吗?", "洗手间在哪里?"]  # Hello | Do you speak English? | Where is the bathroom?
            >>> se.find_similar_sentences(query, cands)
            {
                'query': '欢迎光临',
                'ranking': [(0, '你好。', 0.53), (2, '洗手间在哪里?', 0.2), (1, '你会说英语吗?', 0.09)]
            }

        """
        query_embedding = self._model.encode(query, convert_to_tensor=True)
        corpus_embeddings = self._model.encode(cands, convert_to_tensor=True)

        cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        cos_scores = cos_scores.cpu()

        k = min(len(cos_scores), 5)
        top_results = np.argpartition(-cos_scores, range(k))[0:k]
        top_results = top_results.tolist()

        result = list()
        for idx in top_results:
            result.append(
                (idx, cands[idx].strip(), round(cos_scores[idx].item(), 2)))

        return {
            "query": query.strip(),
            "ranking": result,
        }

    def predict(self, sent: str, **kwargs):
        """
        Conduct sentence embedding

        Args:
            sent (str): input sentence to be sentence embedded

        Returns:
            np.array: embedded sentence array

        """
        outputs = self._model.encode([sent])[0]
        return outputs
