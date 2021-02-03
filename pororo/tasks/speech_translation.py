"""Speech Translation related modeling class"""

from collections import defaultdict
from typing import Optional

from pororo.tasks import PororoFactoryBase, PororoSimpleBase


class PororoSpeechTranslationFactory(PororoFactoryBase):
    """
    Translate source language speech to target language text.
    Currently source language English, Korean and Chinese supports.

    English (`wav2vec|transformer.large.multi.mtpg`)

        - dataset: N/A
        - metric: N/A

    Korean (`wav2vec|transformer.large.multi.mtpg`)

        - dataset: N/A
        - metric: N/A

    Chinese (`wav2vec|transformer.large.multi.mtpg`)

        - dataset: N/A
        - metric: N/A

    Args:
        audio_path (str): audio path for asr (Supports WAV, FLAC and MP3 format.
        tgt (str): target language
        top_db (int): the threshold (in decibels) below reference to consider as silence
        batch_size (int): batch size of input
        vad (bool): flag indication whether to use voice activity or not

    Returns:
         dict: dictionary contains speech recognition outputs and translation outputs.

    Examples:
        >>> st = Pororo(task="st", lang="ko")
        >>> st("korean_speech.wav", tgt="en")
        {
            'asr_output': '카카오 브레인은 대한민국 IT 기업이다.',
            'mt_output': 'Kakao Brain is an IT company in Korea.'
        }

    """

    def __init__(self, task: str, lang: str, model: Optional[str]):
        super().__init__(task, lang, model)

    @staticmethod
    def get_available_langs():
        return ["en", "ko", "zh"]

    @staticmethod
    def get_available_models():
        return {
            "en": ["wav2vec|transformer.large.multi.mtpg"],
            "ko": ["wav2vec|transformer.large.multi.mtpg"],
            "zh": ["wav2vec|transformer.large.multi.mtpg"],
        }

    def load(self, device: str):
        """
        Load user-selected task-specific model

        Args:
            device (str): device information

        Returns:
            object: User-selected task-specific model

        """
        if self.config.lang not in self.get_available_langs():
            raise ValueError(
                f"Unsupported Language : {self.config.lang}",
                'Support Languages : ["ko", "en", "zh"]',
            )
        from pororo.tasks import PororoAsrFactory, PororoTranslationFactory

        asr = PororoAsrFactory(
            task="asr",
            lang=self.config.lang,
            model=f"wav2vec.{self.config.lang}",
        ).load(device)

        mt = PororoTranslationFactory(
            task="mt",
            lang="multi",
            model="transformer.large.multi.mtpg",
        ).load(device)

        return PororoSpeechTranslation(asr, mt, self.config)


class PororoSpeechTranslation(PororoSimpleBase):

    def __init__(self, asr, mt, config):
        super().__init__(config)
        self.src_lang = config.lang

        self.asr = asr
        self.mt = mt

    def predict(
        self,
        audio_path: str,
        tgt: str,
        **kwargs,
    ) -> dict:
        """
        Conduct speech translation on given audio.

        Args:
            audio_path (str): audio path for asr
            tgt (str): target language

        Returns:
             dict: dictionary contains speech recognition outputs and translation outputs.

        """
        top_db = kwargs.get("top_db", 48)
        batch_size = kwargs.get("batch_size", 1)
        vad = kwargs.get("batch_size", False)

        asr_outputs = self.asr(
            audio_path,
            top_db=top_db,
            vad=vad,
            batch_size=batch_size,
        )

        result_dict = defaultdict(list)
        for asr_output in asr_outputs["results"]:
            mt_output = self.mt(
                asr_output["speech"].lower(),
                src=self.src_lang,
                tgt=tgt,
            )

            result_dict["asr_outputs"].append(asr_output)
            result_dict["mt_outputs"].append(mt_output)

        return dict(result_dict)

    def __call__(
        self,
        audio_path: str,
        tgt: str,
        **kwargs,
    ) -> dict:
        return self.predict(
            audio_path,
            tgt,
            **kwargs,
        )
