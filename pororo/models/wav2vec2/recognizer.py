# Copyright (c) Facebook, Inc., its affiliates and Kakao Brain. All Rights Reserved

import datetime
import math
import unicodedata
from typing import Tuple

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from fairseq.data import Dictionary

from pororo.models.vad import VoiceActivityDetection
from pororo.models.wav2vec2.submodules import (
    BrainWav2VecCtc,
    W2lViterbiDecoder,
)
from pororo.models.wav2vec2.utils import collate_fn, get_mask_from_lengths


class BrainWav2Vec2Recognizer(object):
    """ Wav2Vec 2.0 Speech Recognizer """

    graphemes = {
        "ko": [
            "ᅡ", "ᄋ", "ᄀ", "ᅵ", "ᆫ", "ᅳ", "ᅥ", "ᅩ", "ᄂ", "ᄃ", "ᄌ", "ᆯ", "ᄅ",
            "ᄉ", "ᅦ", "ᄆ", "ᄒ", "ᅢ", "ᅮ", "ᆼ", "ᆨ", "ᅧ", "ᄇ", "ᆻ", "ᆷ", "ᅣ",
            "ᄎ", "ᄁ", "ᅯ", "ᄄ", "ᅪ", "ᆭ", "ᆸ", "ᄐ", "ᅬ", "ᄍ", "ᄑ", "ᆺ", "ᇂ",
            "ᅭ", "ᇀ", "ᄏ", "ᅫ", "ᄊ", "ᆹ", "ᅤ", "ᅨ", "ᆽ", "ᄈ", "ᅲ", "ᅱ", "ᇁ",
            "ᅴ", "ᆮ", "ᆩ", "ᆾ", "ᆶ", "ᆰ", "ᆲ", "ᅰ", "ᆱ", "ᆬ", "ᆿ", "ᆴ", "ᆪ", "ᆵ"
        ],
        "en": None,
        "zh": None,
    }

    def __init__(
        self,
        model_path: str,
        dict_path: str,
        device: str,
        lang: str = "en",
        vad_model: VoiceActivityDetection = None,
    ) -> None:
        self.SAMPLE_RATE = 16000
        self.MINIMUM_INPUT_LENGTH = 1024

        self.target_dict = Dictionary.load(dict_path)

        self.lang = lang
        self.graphemes = BrainWav2Vec2Recognizer.graphemes[lang]
        self.device = device

        self.collate_fn = collate_fn
        self.model = self._load_model(model_path, device, self.target_dict)
        self.generator = W2lViterbiDecoder(self.target_dict)
        self.vad_model = vad_model

    def _load_model(self, model_path: str, device: str, target_dict) -> list:
        w2v = torch.load(model_path, map_location=device)
        model = BrainWav2VecCtc.build_model(
            w2v["args"],
            target_dict,
            w2v["pretrain_args"],
        )
        model.load_state_dict(w2v["model"], strict=True)
        model.eval().to(self.device)
        return [model]

    @torch.no_grad()
    def _audio_postprocess(self, feats: torch.FloatTensor) -> torch.FloatTensor:
        if feats.dim == 2:
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        return F.layer_norm(feats, feats.shape)

    def _parse_audio(
        self,
        signal: np.ndarray,
    ) -> Tuple[torch.FloatTensor, float]:
        duration = round(librosa.get_duration(signal, sr=self.SAMPLE_RATE), 2)
        feature = torch.from_numpy(signal).float().to(self.device)
        feature = self._audio_postprocess(feature)
        return feature, duration

    def _grapheme_filter(self, sentence: str) -> str:
        new_sentence = str()
        for item in sentence:
            if item not in self.graphemes:
                new_sentence += item
        return new_sentence

    def _text_postprocess(self, sentence: str) -> str:
        """
        Postprocess model output
        Args:
            sentence (str): naively inferenced sentence from model
        Returns:
            str: post-processed, inferenced sentence
        """
        if self.graphemes:
            # grapheme to character
            sentence = unicodedata.normalize("NFC", sentence.replace(" ", ""))
            sentence = sentence.replace("|", " ").strip()
            return self._grapheme_filter(sentence)

        return sentence.replace(" ", "").replace("|", " ").strip()

    def _split_audio(self, signal: np.ndarray, top_db: int = 48) -> list:
        speech_intervals = list()
        start, end = 0, 0

        non_silence_indices = librosa.effects.split(signal, top_db=top_db)

        for _, end in non_silence_indices:
            speech_intervals.append(signal[start:end])
            start = end

        speech_intervals.append(signal[end:])

        return speech_intervals

    @torch.no_grad()
    def predict(
        self,
        audio_path: str,
        signal: np.ndarray,
        top_db: int = 48,
        vad: bool = False,
        batch_size: int = 1,
    ) -> dict:
        result_dict = dict()

        duration = librosa.get_duration(signal, sr=self.SAMPLE_RATE)
        batch_inference = True if duration > 50.0 else False

        result_dict["audio"] = audio_path
        result_dict["duration"] = str(datetime.timedelta(seconds=duration))
        result_dict["results"] = list()

        if batch_inference:
            if vad:
                speech_intervals = self.vad_model(
                    signal,
                    sample_rate=self.SAMPLE_RATE,
                )
            else:
                speech_intervals = self._split_audio(signal, top_db)

            batches, total_speech_sections, total_durations = self._create_batches(
                speech_intervals,
                batch_size,
            )

            for batch_idx, batch in enumerate(batches):
                net_input, sample = dict(), dict()

                net_input["padding_mask"] = get_mask_from_lengths(
                    inputs=batch["inputs"],
                    seq_lengths=batch["input_lengths"],
                ).to(self.device)
                net_input["source"] = batch["inputs"].to(self.device)
                sample["net_input"] = net_input

                # yapf: disable
                if sample["net_input"]["source"].size(1) < self.MINIMUM_INPUT_LENGTH:
                    continue
                # yapf: enable

                hypos = self.generator.generate(
                    self.model,
                    sample,
                    prefix_tokens=None,
                )

                for hypo_idx, hypo in enumerate(hypos):
                    hypo_dict = dict()
                    hyp_pieces = self.target_dict.string(
                        hypo[0]["tokens"].int().cpu())
                    speech_section = total_speech_sections[batch_idx][hypo_idx]

                    speech_start_time = str(
                        datetime.timedelta(
                            seconds=int(round(
                                speech_section["start"],
                                0,
                            ))))
                    speech_end_time = str(
                        datetime.timedelta(
                            seconds=int(round(
                                speech_section["end"],
                                0,
                            ))))

                    # yapf: disable
                    hypo_dict["speech_section"] = f"{speech_start_time} ~ {speech_end_time}"
                    hypo_dict["length_ms"] = total_durations[batch_idx][hypo_idx] * 1000
                    hypo_dict["speech"] = self._text_postprocess(hyp_pieces)
                    # yapf: enable

                    if hypo_dict["speech"]:
                        result_dict["results"].append(hypo_dict)

                del hypos, net_input, sample

        else:
            net_input, sample, hypo_dict = dict(), dict(), dict()

            feature, duration = self._parse_audio(signal)
            net_input["source"] = feature.unsqueeze(0).to(self.device)

            padding_mask = torch.BoolTensor(
                net_input["source"].size(1)).fill_(False)
            net_input["padding_mask"] = padding_mask.unsqueeze(0).to(
                self.device)

            sample["net_input"] = net_input

            hypo = self.generator.generate(
                self.model,
                sample,
                prefix_tokens=None,
            )
            hyp_pieces = self.target_dict.string(
                hypo[0][0]["tokens"].int().cpu())

            speech_start_time = str(datetime.timedelta(seconds=0))
            speech_end_time = str(
                datetime.timedelta(seconds=int(round(duration, 0))))

            hypo_dict[
                "speech_section"] = f"{speech_start_time} ~ {speech_end_time}"
            hypo_dict["length_ms"] = duration * 1000
            hypo_dict["speech"] = self._text_postprocess(hyp_pieces)

            if hypo_dict["speech"]:
                result_dict["results"].append(hypo_dict)

        return result_dict

    def _create_batches(
        self,
        speech_intervals: list,
        batch_size: int = 1,
    ) -> Tuple[list, list, list]:
        batches = list()
        total_speech_sections = list()
        total_durations = list()

        cumulative_duration = 0
        num_batches = math.ceil(len(speech_intervals) / batch_size)

        for batch_idx in range(num_batches):
            sample = list()
            speech_sections = list()
            durations = list()

            for idx in range(batch_size):
                speech_section = dict()
                speech_intervals_idx = batch_idx * batch_size + idx

                if len(speech_intervals) > speech_intervals_idx:
                    feature, duration = self._parse_audio(
                        speech_intervals[speech_intervals_idx])

                    speech_section["start"] = cumulative_duration
                    cumulative_duration += duration
                    speech_section["end"] = cumulative_duration

                    speech_sections.append(speech_section)
                    sample.append(feature)
                    durations.append(duration)
                else:
                    speech_section["start"] = cumulative_duration
                    speech_section["end"] = cumulative_duration

                    speech_sections.append(speech_section)
                    sample.append(torch.zeros(10))
                    durations.append(0)

            batch = self.collate_fn(sample, batch_size)

            batches.append(batch)
            total_speech_sections.append(speech_sections)
            total_durations.append(durations)

        return batches, total_speech_sections, total_durations
