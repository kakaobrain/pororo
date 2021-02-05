"""Word Sense Disambiguation related modeling class"""

import pickle
from collections import namedtuple
from typing import List, Optional, Tuple, Union

from pororo.tasks.utils.base import (
    PororoFactoryBase,
    PororoTaskGenerationBase,
)
from pororo.tasks.utils.download_utils import download_or_load


class PororoWsdFactory(PororoFactoryBase):
    """
    Conduct Word Sense Disambiguation

    Korean (`transformer.large.ko.wsd`)
        - dataset: https://corpus.korean.go.kr/ 어휘 의미 분석 말뭉치
        - metric: TBU

    Args:
        text (str): sentence to be inputted

    Returns:
        List[Tuple[str, str]]: list of token and its disambiguated meaning tuple

    Examples:
        >>> wsd = Pororo(task="wsd", lang="ko")
        >>> wsd("머리에 이가 있나봐.")
        [detail(morph='머리', pos='NNG', sense_id='01', original=None, meaning='사람이나 동물의 목 위의 부분', english='head'),
        detail(morph='에', pos='JKB', sense_id=None, original=None, meaning=None, english=None),
        detail(morph='▁', pos='SPACE', sense_id=None, original=None, meaning=None, english=None),
        detail(morph='이', pos='NNG', sense_id='01', original=None, meaning='이목의 곤충을 통틀어 이르는 말', english='louse'),
        detail(morph='가', pos='JKS', sense_id=None, original=None, meaning=None, english=None),
        detail(morph='▁', pos='SPACE', sense_id=None, original=None, meaning=None, english=None),
        detail(morph='있', pos='VA', sense_id='01', original=None, meaning='사람이나 동물이 어느 곳에서 떠나거나 벗어나지 아니하고 머물다', english='be; stay'),
        detail(morph='나', pos='EC', sense_id=None, original=None, meaning=None, english=None),
        detail(morph='보', pos='VX', sense_id=None, original=None, meaning=None, english=None),
        detail(morph='아', pos='EF', sense_id=None, original=None, meaning=None, english=None),
        detail(morph='.', pos='SF', sense_id=None, original=None, meaning=None, english=None)]

    """

    def __init__(self, task: str, lang: str, model: Optional[str]):
        super().__init__(task, lang, model)

    @staticmethod
    def get_available_langs():
        return ["ko"]

    @staticmethod
    def get_available_models():
        return {
            "ko": ["transformer.large.ko.wsd"],
        }

    def load(self, device: str):
        """
        Load user-selected task-specific model
        Args:
            device (str): device information
        Returns:
            object: User-selected task-specific model
        """
        from pororo.tasks import PororoPosFactory

        if "transformer.large" in self.config.n_model:
            from fairseq.models.transformer import TransformerModel

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

            morph2idx = pickle.load(
                open(
                    download_or_load(
                        f"misc/morph2idx.{self.config.lang}.pkl",
                        self.config.lang,
                    ),
                    "rb",
                ))
            tag2idx = pickle.load(
                open(
                    download_or_load(
                        f"misc/tag2idx.{self.config.lang}.pkl",
                        self.config.lang,
                    ),
                    "rb",
                ))
            query2origin, query2meaning, query2eng, _ = pickle.load(
                open(
                    download_or_load(
                        f"misc/wsd-dicts.{self.config.lang}.pkl",
                        self.config.lang,
                    ),
                    "rb",
                ))

            return PororoTransformerWsd(
                model,
                morph2idx,
                tag2idx,
                query2origin,
                query2meaning,
                query2eng,
                self.config,
            )


class PororoTransformerWsd(PororoTaskGenerationBase):

    def __init__(
        self,
        model,
        morph2idx,
        tag2idx,
        query2origin,
        query2meaning,
        query2eng,
        config,
    ):
        super().__init__(config)
        self._model = model
        self._cands = [
            "NNG", "NNB", "NNBC", "VV", "VA", "MM", "MAG", "NP", "NNP"
        ]

        self._morph2idx = morph2idx
        self._tag2idx = tag2idx
        self._query2origin = query2origin
        self._query2meaning = query2meaning
        self._query2eng = query2eng
        self._Wdetail = namedtuple(
            "detail",
            "morph pos sense_id original meaning english",
        )

    def _preprocess(self, text: str) -> str:
        """
        Preprocess input sentence to replace whitespace token with whitespace

        Args:
            text (str): input sentence

        Returns:
            str: preprocessed input sentence

        """
        text = text.replace(" ", "▁")
        return " ".join([c for c in text])

    def _postprocess(self, output):
        eojeols = output.split("▁")

        result = []
        for i, eojeol in enumerate(eojeols):
            pairs = eojeol.split("++")

            for pair in pairs:
                morph, tag = pair.strip().split(" || ")

                if "__" in morph:
                    morph, sense = morph.split(" __ ")
                else:
                    sense = None

                morph = "".join([c for c in morph if c != " "])

                if tag not in self._cands:
                    result.append(
                        self._Wdetail(
                            morph,
                            tag,
                            None,
                            None,
                            None,
                            None,
                        ))
                    continue

                sense = str(sense).zfill(2)

                query_morph = morph + "다" if tag[0] == "V" else morph
                query_sense = "" if sense is None else "__" + sense

                query = f"{query_morph}{query_sense}"
                origin = self._query2origin.get(query, None)
                meaning = self._query2meaning.get(query, None)
                eng = self._query2eng.get(query, None)

                if meaning is None:
                    query = f"{query_morph}__00"
                    meaning = self._query2meaning.get(query, None)
                    if meaning:
                        sense = "00"

                result.append(
                    self._Wdetail(
                        morph,
                        tag,
                        sense,
                        origin,
                        meaning,
                        eng,
                    ))

            if i != len(eojeols) - 1:
                result.append(
                    self._Wdetail(
                        "▁",
                        "SPACE",
                        None,
                        None,
                        None,
                        None,
                    ))

        return result

    def predict(
        self,
        text: str,
        beam: int,
        **kwargs,
    ) -> Union[List[Tuple[str, str]], None]:
        """
        Conduct Word Sense Disambiguation

        Args:
            text (str): sentence to be inputted
            beam (int): beam search argument
            ignore_none (bool): whether to ignore `none` meaning

        Returns:
            List[Tuple[str]]: list of token and its disambiguated information tuple

        """
        ignore_none = kwargs.get("ignore_none", False)

        text = self._preprocess(text)

        output = self._model.translate(
            text,
            beam=beam,
            max_len_a=4,
            max_len_b=50,
        )
        try:
            result = self._postprocess(output)
            return ([pair for pair in result if pair.meaning is not None]
                    if ignore_none else result)
        except:
            print("Invalid inference result !")
