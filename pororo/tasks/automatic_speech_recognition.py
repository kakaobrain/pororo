"""Automatic Speech Recognition related modeling class"""

import logging
import os
import numpy as np
from typing import Optional
from pororo.tasks.utils.base import PororoFactoryBase, PororoSimpleBase
from pororo.tasks.utils.download_utils import download_or_load


class PororoAsrFactory(PororoFactoryBase):
    """
    Recognized speech sentence using trained model.
    Currently English, Korean and Chinese supports.

    English (`wav2vec.en`)

        - dataset: LibriSpeech
        - metric: WER (clean: 1.9 / other: 4.3)

    Korean (`wav2vec.ko`)

        - dataset: KsponSpeech
        - metric: CER (clean: 4.9 / other: 5.4)

    Chinese (`wav2vec.zh`)

        - dataset: AISHELL-1
        - metric: CER (6.9)

    Args:
        audio_path (str): audio path for asr or youtube link (Supports WAV, FLAC and MP3 format)
        top_db (int): the threshold (in decibels) below reference to consider as silence
        vad (bool): flag indication whether to use voice activity detection or not, If it is False, it is split into
             dB criteria and then speech recognition is made. Applies only when audio length is more than 50 seconds.
        batch_size (int): inference batch size

    Returns:
        dict: result of speech recognition

    Examples:
        >>> asr = Pororo('asr', lang='ko')
        >>> asr('korean_speech.wav')
        {
            'audio': 'example.wav',
            'duration': '0:00:03.297250',
            'results': [
                {
                    'speech_section': '0:00:00 ~ 0:00:03',
                    'length_ms': 3300.0,
                     speech': '이 책은 살 만한 가치가 없어'
                }
            ]
        }
        >>> asr = Pororo('asr', lang='en')
        >>> asr('english_speech.wav')
        {
            'audio': 'english_speech.flac',
            'duration': '0:00:12.195000',
            'results': [
                {
                    'speech_section': '0:00:00 ~ 0:00:12',
                    'length_ms': 12200.0,
                    'speech': 'WELL TOO IF HE LIKE LOVE WOULD FILCH OUR HOARD WITH PLEASURE TO OURSELVES SLUICING
                               OUR VEIN AND VIGOUR TO PERPETUATE THE STRAIN OF LIFE BY SPILTH OF LIFE WITHIN US STORED'
                }
            ]
        }
        >>> asr = Pororo('asr', lang='ko')
        >>> asr('https://www.youtube.com/watch?v=5q9zIgylu1E')
        {
            'audio': '5q9zIgylu1E.wav',
            'duration': '0:18:38.058687',
            'results': [
                {
                    ...
                    ...
                    ...
                }
            ]
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
            "en": ["wav2vec.en"],
            "ko": ["wav2vec.ko"],
            "zh": ["wav2vec.zh"],
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
        from pororo.models.wav2vec2.recognizer import BrainWav2Vec2Recognizer

        model_path = download_or_load(
            f"misc/{self.config.n_model}.pt",
            self.config.lang,
        )
        dict_path = download_or_load(
            f"misc/{self.config.lang}.ltr.txt",
            self.config.lang,
        )
        vad_model_path = download_or_load(
            "misc/vad.pt",
            lang="multi",
        )

        try:
            import librosa
            logging.getLogger("librosa").setLevel(logging.WARN)
        except ModuleNotFoundError as error:
            raise error.__class__(
                "Please install librosa with: `pip install librosa`")

        from pororo.models.vad import VoiceActivityDetection

        vad_model = VoiceActivityDetection(
            model_path=vad_model_path,
            device=device,
        )

        model = BrainWav2Vec2Recognizer(
            model_path=model_path,
            dict_path=dict_path,
            vad_model=vad_model,
            device=device,
            lang=self.config.lang,
        )
        return PororoASR(model, self.config)


class PororoASR(PororoSimpleBase):

    def __init__(self, model, config):
        super().__init__(config)
        self._model = model
        self.SAMPLE_RATE = 16000
        self.MAX_VALUE = 32767

    def _download_audio_from_youtube(self, url: str, filename: str) -> bool:
        """
        Download audio file from youtube link

        Args:
            url: youtube link
            filename: save audio file name

        Returns:
            bool: return success or failure.

        """
        try:
            import youtube_dl
        except ImportError:
            raise ImportError(
                "Please install youtube_dl: https://github.com/ytdl-org/youtube-dl"
            )
        try:
            import soundfile as sf
        except ImportError:
            raise ImportError("Please install soundfile: pip install soundfile")
        try:
            import librosa
        except ImportError:
            raise ImportError("Please install librosa: pip install librosa")

        ydl_opts = {
            "format":
                "bestaudio/best",
            "outtmpl":
                f"{filename}.wav",
            "noplaylist":
                True,
            "continue_dl":
                True,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }],
        }

        if os.path.isfile(f"{filename}.wav"):
            os.remove(f"{filename}.wav")

        try:
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.cache.remove()
                info_dict = ydl.extract_info(url, download=False)
                ydl.prepare_filename(info_dict)
                ydl.download([url])

        except Exception:
            raise AssertionError("Audio download from youtube failed")

        # Re-write to match the format required by Pydub
        sample_rate = librosa.get_samplerate(f"{filename}.wav")
        signal, _ = librosa.load(f"{filename}.wav", sr=sample_rate)
        sf.write(f"{filename}.wav", data=signal, samplerate=sample_rate)
        return True

    def _youtube_link_predict(
        self,
        audio_path: str,
        vad: bool,
        video_id: str,
        top_db: int = 48,
        batch_size: int = 1,
    ) -> dict:
        """
        Conduct speech recognition for audio from youtube link.

        Args:
            audio_path (str): the wav file path
            top_db (int): the threshold (in decibels) below reference to consider as silence
            batch_size (int): inference batch size
            vad (bool): flag indication whether to use voice activity detection or not, If it is False, it is split into
             dB criteria and then speech recognition is made. Applies only when audio length is more than 50 seconds.
            video_id (str): video unique ID value

        Returns:
            dict: result of speech recognition

        """

        assert self._download_audio_from_youtube(
            audio_path, filename=video_id), "download from youtube failed"
        signal = self._preprocess_audio(f"{video_id}.wav")
        os.remove(f"{video_id}.wav")

        return self._model.predict(
            audio_path=audio_path,
            signal=signal,
            vad=vad,
            batch_size=batch_size,
            top_db=top_db,
        )

    def _preprocess_audio(self, audio_path: str):
        try:
            import librosa
        except ImportError:
            raise ImportError("Please install librosa: pip install librosa")
        try:
            # Using the pydub because the speed of the resample is the fastest.
            from pydub import AudioSegment
        except ImportError:
            raise ImportError("Please install pydub: pip install pydub")

        audio_extension = audio_path.split('.')[-1].lower()
        assert audio_extension in (
            'wav', 'mp3', 'flac',
            'pcm'), f"Unsupported format: {audio_extension}"

        if audio_extension == 'pcm':
            signal = np.memmap(
                audio_path,
                dtype='h',
                mode='r',
            ).astype('float32')

        else:
            sample_rate = librosa.get_samplerate(audio_path)
            signal = AudioSegment.from_file(
                audio_path,
                format=audio_extension,
                frame_rate=sample_rate,
            )

            if sample_rate != self.SAMPLE_RATE:
                signal = signal.set_frame_rate(frame_rate=self.SAMPLE_RATE)

            channel_sounds = signal.split_to_mono()
            signal = np.array(
                [s.get_array_of_samples() for s in channel_sounds])[0]

        return signal / self.MAX_VALUE

    def predict(
        self,
        audio_path: str,
        top_db: int = 48,
        batch_size: int = 1,
        vad: bool = False,
    ) -> dict:
        """
        Conduct speech recognition for audio in a given path

        Args:
            audio_path (str): the wav file path
            top_db (int): the threshold (in decibels) below reference to consider as silence (default: 48)
            batch_size (int): inference batch size (default: 1)
            vad (bool): flag indication whether to use voice activity detection or not, If it is False, it is split into
             dB criteria and then speech recognition is made. Applies only when audio length is more than 50 seconds.

        Returns:
            dict: result of speech recognition

        """
        if "youtu.be/" in audio_path:
            video_id = audio_path.split("outu.be/")[1][:11]
            audio_path = f"https://www.youtube.com/watch?v={video_id}"

        if "youtube.com/watch?v=" in audio_path:
            video_id = audio_path.split("outube.com/watch?v=")[1][:11]
            return self._youtube_link_predict(
                audio_path=audio_path,
                vad=vad,
                video_id=video_id,
                top_db=top_db,
                batch_size=batch_size,
            )

        signal = self._preprocess_audio(audio_path)

        return self._model.predict(
            audio_path=audio_path,
            signal=signal,
            top_db=top_db,
            vad=vad,
            batch_size=batch_size,
        )

    def __call__(
        self,
        audio_path: str,
        top_db: int = 48,
        batch_size: int = 16,
        vad: bool = False,
    ) -> dict:
        return self.predict(
            audio_path=audio_path,
            top_db=top_db,
            batch_size=batch_size,
            vad=vad,
        )
