"""Semantic Role Labeling related modeling class"""

from copy import deepcopy
from typing import List, Optional

from pororo.tasks.utils.base import PororoFactoryBase, PororoSimpleBase


class PororoSrlFactory(PororoFactoryBase):
    """
    Conduct semantic role labeling

    Korean (`charbert.base.ko.srl`)

        - dataset: UCorpus
        - metric: TBU
        - ref: http://nlplab.ulsan.ac.kr/doku.php?id=start

    Args:
        sent: (str) sentence to be parsed dependency

    Returns:
        List[Tuple[int, str, int, str]]: token index, token label, token head and its relation

    Examples:
        >>> srl = Pororo(task="srl", lang="ko")
        >>> srl("카터는 역삼에서 카카오브레인으로 출근한다.")
        [[('카터는', 'AGT'), ('역삼에서', 'LOC'), ('카카오브레인으로', 'GOL'), ('출근한다.', 'PREDICATE')]]
        >>> srl("피고인은 거제에서 400만 원 상당의 순금목걸이를 피해자로부터 강취하였다.")
        [[('피고인은', 'AGT'), ('거제에서', '-'), ('400만', '-'), ('원', '-'), ('상당의', '-'), ('순금목걸이를', 'THM'), ('피해자로부터', 'SRC'), ('강취하였다.', 'PREDICATE')]]

    """

    def __init__(self, task: str, lang: str, model: Optional[str]):
        super().__init__(task, lang, model)

    @staticmethod
    def get_available_langs():
        return ["ko"]

    @staticmethod
    def get_available_models():
        return {"ko": ["charbert.base.ko.srl"]}

    def load(self, device: str):
        """
        Load user-selected task-specific model

        Args:
            device (str): device information

        Returns:
            object: User-selected task-specific model

        """
        if "charbert" in self.config.n_model:
            from pororo.models.brainbert import RobertaLabelModel
            from pororo.tasks import PororoPosFactory

            model = RobertaLabelModel.load_model(
                f"bert/{self.config.n_model}",
                self.config.lang,
            ).eval().to(device)

            tagger = PororoPosFactory(
                task="pos",
                model="mecab-ko",
                lang=self.config.lang,
            ).load(device)

            return PororoBertSRL(model, tagger, self.config)


class PororoBertSRL(PororoSimpleBase):

    def __init__(self, model, tagger, config):
        super().__init__(config)
        self._tagger = tagger
        self._model = model
        self._verbs = ["VV", "VA", "XSV", "XSA", "VCN"]

    def _split_list(self, lst: List, seperator: str):
        """
        Split list using seperator

        Args:
            lst (list): PoS tagger pair list
            seperator (str): seperator token

        Returns:
            list: splitted list of list

        """
        res = []
        tmp = []
        for elem in lst:
            if elem[0] == seperator:
                res.append(tmp)
                tmp = []
                continue
            tmp.append(elem)
        res.append(tmp)
        return res

    def _preprocess(self, sent: str) -> str:
        """
        Preprocess semantic role labeling input to specify predicate

        Args:
            sent (str): input sentence

        Returns:
            str: preprocessed input

        """
        words = self._split_list([list(tag) for tag in self._tagger(sent)], " ")

        vs = []
        for i, word in enumerate(words):
            for morph in word:
                if morph[1] in self._verbs:
                    vs.append(i)
                    break

        sents = []
        for v in vs:
            morphs = deepcopy(words)
            morphs[v][0][0] = f"★{morphs[v][0][0]}"

            sent, seg = str(), str()

            for elems in morphs:
                for pair in elems:
                    morph, tag = pair
                    tag = f"{tag} "
                    if morph == " ":
                        sent += "▁ "
                        seg += tag
                        continue

                    chars = [c for c in morph]
                    sent += f"{' '.join(chars)} "
                    seg += tag * len(chars)

                sent += "▁ "
                seg += "SPACE "
            sents.append((sent.strip(), seg.strip()))

        return sents

    def _postprocess(self, result: List, origin: str):
        """
        Postprocess semantic role labeling model inference result

        Args:
            result (List): inferenced semantic roles
            origin (str): original query string

        Returns:
            List[Tuple]: postprocessed result

        """
        tokens = origin.split()
        fin = []
        for res in result:
            res = self._split_list(res, "▁")
            tmp = []
            for i, token in enumerate(tokens):
                if "★" in res[i][0][0]:
                    tmp.append((token, "PREDICATE"))
                    continue
                tmp.append((token, res[i][0][1]))
            fin.append(tmp)
        return fin

    def predict(self, sent: str, **kwargs):
        """
        Conduct semantic role labeling

        Args:
            sent: (str) sentence to be parsed dependency

        Returns:
            List[Tuple[int, str, int, str]]: token index, token label, token head and its relation

        """
        preproc = self._preprocess(sent)

        if not preproc:
            return "There is NO predicate to be labeled"

        res = []
        for p in preproc:
            res.append(self._model.predict_srl(p[0], p[1]))
        return self._postprocess(res, sent)
