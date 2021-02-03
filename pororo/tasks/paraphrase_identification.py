"""Paraphrase Identification related modeling class"""

from typing import Optional

from pororo.tasks.utils.base import PororoBiencoderBase, PororoFactoryBase


class PororoParaIdFactory(PororoFactoryBase):
    """
    Classification based paraphrase identification

    Korean (`brainbert.base.ko.paws`)

        - dataset: PAWS-X (Yinfei Yang et al. 2019)
        - metric: Accuracy (83.75)

    Examples:
        >>> paws("그는 빨간 자전거를 샀다", "그가 산 자전거는 빨간색이다.")
        'Paraphrase'
        >>> paws("그는 빨간 자전거를 샀다", "그가 타고 있는 자전거는 빨간색이다.")
        'NOT Paraphrase'
        >>> paws("그녀는 제주도에서 일출을 감상했다", "그녀는 일출을 감상하기 위해서 제주도에 갔다.")
        'Paraphrase'
        >>> paws("그녀는 제주도에서 일출을 감상했다", "그녀는 제주도에 갔다.")
        'Paraphrase'
        >>> paws("그녀는 제주도에서 일출을 감상했다", "그녀는 일출을 감상했다")
        'Paraphrase'
        >>> paws("그녀는 제주도에서 일출을 감상했다", "그녀는 강릉에서 일출을 감상했다")
        'NOT Paraphrase'

    """

    def __init__(self, task: str, lang: str, model: Optional[str]):
        super().__init__(task, lang, model)

    @staticmethod
    def get_available_langs():
        return ["ko"]

    @staticmethod
    def get_available_models():
        return {"ko": ["brainbert.base.ko.paws"]}

    def load(self, device: str):
        """
        Load user-selected task-specific model

        Args:
            device (str): device information

        Returns:
            object: User-selected task-specific model

        """
        if "brainbert" in self.config.n_model:
            from pororo.models.brainbert import BrainRobertaModel

            model = (BrainRobertaModel.load_model(
                f"bert/{self.config.n_model}",
                self.config.lang,
            ).eval().to(device))
            return PororoBertParaId(model, self.config)


class PororoBertParaId(PororoBiencoderBase):

    def __init__(self, model, config):
        super().__init__(config)
        self._model = model
        self._label_fn = {
            "0": "NOT Paraphrase",
            "1": "Paraphrase",
        }

    def predict(self, sent_a: str, sent_b: str, **kwargs):
        """
        Conduct paraphrase identification

        Args:
            sent_a (str): first sentence to be encoded
            sent_b (str): second sentence to be encoded

        Returns:
            str: paraphrase identified result - `Not Paraphrase` or `Paraphrase`

        """
        return self._label_fn[self._model.predict_output(sent_a, sent_b)]
