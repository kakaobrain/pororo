import numpy as np
import torch

from pororo.models.tts.utils.dsp import *
from pororo.models.tts.waveRNN.params import hp


def generate(model, spectrogram, batched, target, overlap, save_str=None):
    mel = normalize(spectrogram)
    if mel.ndim != 2 or mel.shape[0] != hp.num_mels:
        raise ValueError("Expected a numpy array shaped (n_mels, n_hops) !")
    _max = np.max(mel)
    _min = np.min(mel)
    if _max >= 1.01 or _min <= -0.01:
        raise ValueError(
            f"Expected spectrogram range in [0,1] but was instead [{_min}, {_max}]"
        )
    mel = torch.tensor(mel).unsqueeze(0)
    return model.generate(mel, save_str, batched, target, overlap, hp.mu_law)
