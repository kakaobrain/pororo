import numpy as np
import scipy

try:
    import librosa  # noqa
    import librosa.effects  # noqa
    import librosa.feature  # noqa
except ImportError:
    raise ImportError("Please install librosa with: `pip install librosa`")
import soundfile as sf

try:
    from fastdtw import fastdtw  # noqa
except ImportError:
    raise ImportError("Please install fastdtw with: `pip install fastdtw`")
from pororo.models.tts.tacotron.params import Params as hp


def load(path):
    """Load a sound file into numpy array."""
    data, sample_rate = sf.read(path)
    assert (
        hp.sample_rate == sample_rate
    ), f"Sample rate do not match: given {hp.sample_rate}, expected {sample_rate}"
    return data


def save(data, path):
    """Save numpy array as sound file."""
    sf.write(path, data, samplerate=hp.sample_rate)


def ms_to_frames(ms):
    """Convert milliseconds into number of frames."""
    return int(hp.sample_rate * ms / 1000)


def trim_silence(data, window_ms, hop_ms, top_db=50, margin_ms=0):
    """Trim leading and trailing silence from an audio signal."""
    wf = ms_to_frames(window_ms)
    hf = ms_to_frames(hop_ms)
    mf = ms_to_frames(margin_ms)
    if mf != 0:
        data = data[mf:-mf]
    return librosa.effects.trim(data,
                                top_db=top_db,
                                frame_length=wf,
                                hop_length=hf)


def duration(data):
    """Return duration of an audio signal in seconds."""
    return librosa.get_duration(data, sr=hp.sample_rate)


def amplitude_to_db(x):
    """Convert amplitude to decibels."""
    return librosa.amplitude_to_db(x, ref=np.max, top_db=None)


def db_to_amplitude(x):
    """Convert decibels to amplitude."""
    return librosa.db_to_amplitude(x)


def preemphasis(y):
    """Preemphasize the signal."""
    # y[n] = x[n] - perc * x[n-1]
    return scipy.signal.lfilter([1, -hp.preemphasis], [1], y)


def spectrogram(y, mel=False):
    """Convert waveform to log-magnitude spectrogram."""
    if hp.use_preemphasis:
        y = preemphasis(y)
    wf = ms_to_frames(hp.stft_window_ms)
    hf = ms_to_frames(hp.stft_shift_ms)
    S = np.abs(librosa.stft(y, n_fft=hp.num_fft, hop_length=hf, win_length=wf))
    if mel:
        S = librosa.feature.melspectrogram(S=S,
                                           sr=hp.sample_rate,
                                           n_mels=hp.num_mels)
    return amplitude_to_db(S)


def mel_spectrogram(y):
    """Convert waveform to log-mel-spectrogram."""
    return spectrogram(y, True)


def linear_to_mel(S):
    """Convert linear to mel spectrogram (this does not return the same spec. as mel_spec. method due to the db->amplitude conversion)."""
    S = db_to_amplitude(S)
    S = librosa.feature.melspectrogram(S=S,
                                       sr=hp.sample_rate,
                                       n_mels=hp.num_mels)
    return amplitude_to_db(S)


def normalize_spectrogram(S, is_mel):
    """Normalize log-magnitude spectrogram."""
    if is_mel:
        return (S - hp.mel_normalize_mean) / hp.mel_normalize_variance
    else:
        return (S - hp.lin_normalize_mean) / hp.lin_normalize_variance


def denormalize_spectrogram(S, is_mel):
    """Denormalize log-magnitude spectrogram."""
    if is_mel:
        return S * hp.mel_normalize_variance + hp.mel_normalize_mean
    else:
        return S * hp.lin_normalize_variance + hp.lin_normalize_mean
