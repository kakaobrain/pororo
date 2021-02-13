import json
from typing import Tuple

import librosa
import torch

from pororo.models.tts.hifigan.checkpoint import load_checkpoint
from pororo.models.tts.hifigan.model import Generator
from pororo.models.tts.synthesis import synthesize
from pororo.models.tts.tacotron.params import Params as tacotron_hp
from pororo.models.tts.tacotron.tacotron2 import Tacotron
from pororo.models.tts.utils import remove_dataparallel_prefix
from pororo.models.tts.waveRNN.gen_wavernn import generate as wavernn_generate
from pororo.models.tts.waveRNN.params import hp as wavernn_hp
from pororo.models.tts.waveRNN.waveRNN import WaveRNN


class AttrDict(dict):

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class MultilingualSpeechSynthesizer(object):

    def __init__(
        self,
        tacotron_path: str,
        english_vocoder_path: str,
        english_vocoder_config: str,
        korean_vocoder_path: str,
        korean_vocoder_config: str,
        wavernn_path: str,
        device: str,
        lang: str = "en",
    ):
        self.lang = lang
        self.device = device
        self.vocoder_en_config = None
        self.vocoder_ko_config = None

        self.tacotron, self.vocoder_en, self.vocoder_ko, self.vocoder_multi = self.build_model(
            tacotron_path,
            english_vocoder_path,
            english_vocoder_config,
            korean_vocoder_path,
            korean_vocoder_config,
            wavernn_path,
        )

    def _build_hifigan(self, config: str, hifigan_path: str) -> Generator:
        with open(config) as f:
            data = f.read()

        config = json.loads(data)
        config = AttrDict(config)

        generator = Generator(config).to(self.device)
        state_dict_g = load_checkpoint(hifigan_path, self.device)
        generator.load_state_dict(state_dict_g['generator'])
        generator.eval()
        generator.remove_weight_norm()

        return generator

    def _build_tacotron(self, tacotron_path: str) -> Tacotron:
        state = torch.load(tacotron_path, map_location=self.device)
        tacotron_hp.load_state_dict(state["parameters"])
        tacotron = Tacotron()
        tacotron.load_state_dict(remove_dataparallel_prefix(state["model"]))
        tacotron.eval().to(self.device)
        return tacotron

    def _build_wavernn(self, wavernn_path: str) -> WaveRNN:
        wavernn = (WaveRNN(
            rnn_dims=wavernn_hp.voc_rnn_dims,
            fc_dims=wavernn_hp.voc_fc_dims,
            bits=wavernn_hp.bits,
            pad=wavernn_hp.voc_pad,
            upsample_factors=wavernn_hp.voc_upsample_factors,
            feat_dims=wavernn_hp.num_mels,
            compute_dims=wavernn_hp.voc_compute_dims,
            res_out_dims=wavernn_hp.voc_res_out_dims,
            res_blocks=wavernn_hp.voc_res_blocks,
            hop_length=wavernn_hp.hop_length,
            sample_rate=wavernn_hp.sample_rate,
            mode=wavernn_hp.voc_mode,
        ).eval().to(self.device))
        wavernn.load(wavernn_path)
        return wavernn

    def build_model(
        self,
        tacotron_path: str,
        english_vocoder_path: str,
        english_vocoder_config: str,
        korean_vocoder_path: str,
        korean_vocoder_config: str,
        wavernn_path: str,
    ) -> Tuple[Tacotron, Generator, Generator, WaveRNN]:
        """Load and build tacotron a from checkpoint."""
        tacotron = self._build_tacotron(tacotron_path)
        vocoder_multi = self._build_wavernn(wavernn_path)
        vocoder_ko = self._build_hifigan(
            korean_vocoder_config,
            korean_vocoder_path,
        )
        vocoder_en = self._build_hifigan(
            english_vocoder_config,
            english_vocoder_path,
        )
        return tacotron, vocoder_en, vocoder_ko, vocoder_multi

    def _spectrogram_postprocess(self, spectrogram):
        spectrogram = librosa.db_to_amplitude(spectrogram)
        spectrogram = torch.log(
            torch.clamp(torch.Tensor(spectrogram), min=1e-5) * 1)
        return spectrogram

    def predict(self, text: str, speaker: str):
        speakers = speaker.split(',')
        spectrogram = synthesize(self.tacotron, f"|{text}", device=self.device)

        if len(speakers) > 1:
            audio = wavernn_generate(
                self.vocoder_multi,
                spectrogram,
                wavernn_hp.voc_gen_batched,
                wavernn_hp.voc_target,
                wavernn_hp.voc_overlap,
            )
            audio = audio * 32768.0
            return audio

        if speaker in ("ko", "en"):
            spectrogram = self._spectrogram_postprocess(spectrogram)

            if speaker == "ko":
                y_g_hat = self.vocoder_ko(
                    torch.Tensor(spectrogram).to(self.device).unsqueeze(0))
            else:
                y_g_hat = self.vocoder_en(
                    torch.Tensor(spectrogram).to(self.device).unsqueeze(0))

            audio = y_g_hat.squeeze()
            audio = audio * 32768.0
            return audio.cpu().detach().numpy()

        else:
            audio = wavernn_generate(
                self.vocoder_multi,
                spectrogram,
                wavernn_hp.voc_gen_batched,
                wavernn_hp.voc_target,
                wavernn_hp.voc_overlap,
            )
            audio = audio * 32768.0
            return audio
