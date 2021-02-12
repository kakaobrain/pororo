"""Speech Synthesis related modeling class"""

from typing import Optional, Tuple
from numpy import ndarray

from pororo.tasks import (
    PororoFactoryBase,
    PororoG2pFactory,
    PororoSimpleBase,
    download_or_load,
)


class PororoTtsFactory(PororoFactoryBase):
    """
    Synthesis text to speech using trained model.
    Output audio's sample rate is 22050.

    Multi (`tacotron`)

        - dataset: TBU
        - metric: TBU

    Args:
        text (str): text for speech synthesis
        lang (str): text's language Ex) how are you?: en, 안녕하세요.: ko
        speaker (str): designate a speaker such as ko, en, zh etc.. (default: lang)

    Returns:
        ndarray: waveform of speech signal

    Examples:
        >>> import IPython
        >>> from pororo import Pororo
        >>> model = Pororo('tts', lang='multi')

        >>> # Typical TTS
        >>> wave = model('how are you?', lang='en')
        >>> IPython.display.Audio(data=wave, rate=22050)

        >>> # Voice Style Transfer
        >>> wave = model('저는 미국 사람이에요.', lang='ko', speaker='en')
        >>> IPython.display.Audio(data=wave, rate=22050)

        >>> # Code-Switching
        >>> wave = model('저는 미국 사람이에요.', lang='ko', speaker='en-15,ko')
        >>> IPython.display.Audio(data=wave, rate=22050)

    Notes:
        Currently 11 languages supports.
        Supported Languages: English, Korean, Japanese, Chinese, Jejueo, Dutch, German, Spanish, French, Russian, Finnish
        This task can designate a speaker such as ko, en, zh etc.

    """

    def __init__(self, task: str, lang: str = "multi", model: Optional[str] = None):
        super().__init__(task, lang, model)

    @staticmethod
    def get_available_langs():
        return ["multi"]

    @staticmethod
    def get_available_models():
        return {
            "multi": ["tacotron"],
        }

    def load(self, device: str):
        """
        Load user-selected task-specific model

        Args:
            device (str): device information

        Returns:
            object: User-selected task-specific model

        """
        if self.config.n_model == "tacotron":
            from pororo.models.tts.synthesizer import MultilingualSpeechSynthesizer
            from pororo.models.tts.utils.numerical_pinyin_converter import convert_from_numerical_pinyin
            from pororo.models.tts.utils.text import jejueo_romanize, romanize

            tacotron_path = download_or_load("misc/tacotron2", self.config.lang)
            english_vocoder_path = download_or_load("misc/hifigan_en", self.config.lang)
            korean_vocoder_path = download_or_load("misc/hifigan_ko", self.config.lang)
            english_vocoder_config = download_or_load("misc/hifigan_en_config.json", self.config.lang)
            korean_vocoder_config = download_or_load("misc/hifigan_ko_config.json", self.config.lang)
            wavernn_path = download_or_load("misc/wavernn.pyt", self.config.lang)
            synthesizer = MultilingualSpeechSynthesizer(
                tacotron_path,
                english_vocoder_path,
                english_vocoder_config,
                korean_vocoder_path,
                korean_vocoder_config,
                wavernn_path,
                device,
                self.config.lang,
            )
            return PororoTTS(
                synthesizer,
                device,
                romanize,
                jejueo_romanize,
                convert_from_numerical_pinyin,
                self.config,
            )


class PororoTTS(PororoSimpleBase):

    def __init__(
        self,
        synthesizer,
        device,
        romanize,
        jejueo_romanize,
        convert_from_numerical_pinyin,
        config,
    ):
        super().__init__(config)
        self._synthesizer = synthesizer

        self.g2p_ja = None
        self.g2p_zh = None

        self.lang_dict = {
            "en": "en",
            "ko": "ko",
            "ja": "jp",
            "de": "de",
            "nl": "nl",
            "ru": "ru",
            "es": "es",
            "fr": "fr",
            "zh": "zh",
            "fi": "fi",
            "je": "je",
        }
        self.device = device

        self.romanize = romanize
        self.jejueo_romanize = jejueo_romanize
        self.convert_from_numerical_pinyin = convert_from_numerical_pinyin

    def _load_g2p_ja(self):
        """ load g2p module for Japanese """
        self.g2p_ja = PororoG2pFactory(
            task="g2p",
            model="g2p.ja",
            lang="ja",
        )
        self.g2p_ja = self.g2p_ja.load(self.device)

    def _load_g2p_zh(self):
        """ load g2p module for Chinese """
        self.g2p_zh = PororoG2pFactory(
            task="g2p",
            model="g2p.zh",
            lang="zh",
        )
        self.g2p_zh = self.g2p_zh.load(self.device)

    def _preprocess(
        self,
        text: str,
        lang: str = "en",
        speaker: str = None,
    ) -> Tuple[str, str]:
        """
        Pre-process text for TTS format

        Args:
            text (str): text for tts
            lang (str): text language
            speaker (speaker): designation of speaker

        Returns:
            str: pre-processed text

        """
        if lang == "ko":
            text = self.romanize(text)
        elif lang == "ja":
            if self.g2p_ja is None:
                self._load_g2p_ja()
            text = self.g2p_ja(text)
        elif lang == "zh":
            if self.g2p_zh is None:
                self._load_g2p_zh()
            text = self.g2p_zh(text).replace("   ", " ")
            text = self.convert_from_numerical_pinyin(text)
        elif lang == "je":
            text = self.jejueo_romanize(text)
        return f"{text}|00-{self.lang_dict[lang]}|{speaker}", speaker

    def predict(self, text: str, speaker: str) -> ndarray:
        """
        Conduct speech synthesis on given text

        Args:
            text (str): text for tts
            speaker (speaker): designation of speaker

        Returns:
             ndarray: waveform of speech signal

        """
        return self._synthesizer.predict(text, speaker)

    def __call__(self, text: str, lang: str = "en", speaker: str = None):
        if speaker is None:
            speaker = lang

        text, speaker = self._preprocess(text, lang, speaker)
        return self.predict(text, speaker)
