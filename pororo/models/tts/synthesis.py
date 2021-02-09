import torch

from pororo.models.tts.tacotron.params import Params as hp
from pororo.models.tts.utils import audio, text


def synthesize(model, input_data, force_cpu=False, device=None):
    item = input_data.split("|")
    clean_text = item[1]

    if not hp.use_punctuation:
        clean_text = text.remove_punctuation(clean_text)
    if not hp.case_sensitive:
        clean_text = text.to_lower(clean_text)
    if hp.remove_multiple_wspaces:
        clean_text = text.remove_odd_whitespaces(clean_text)

    t = torch.LongTensor(
        text.to_sequence(clean_text, use_phonemes=hp.use_phonemes))

    if hp.multi_language:
        l_tokens = item[3].split(",")
        t_length = len(clean_text) + 1
        l = []
        for token in l_tokens:
            l_d = token.split("-")

            language = [0] * hp.language_number
            for l_cw in l_d[0].split(":"):
                l_cw_s = l_cw.split("*")
                language[hp.languages.index(
                    l_cw_s[0])] = (1 if len(l_cw_s) == 1 else float(l_cw_s[1]))

            language_length = int(l_d[1]) if len(l_d) == 2 else t_length
            l += [language] * language_length
            t_length -= language_length
        l = torch.FloatTensor([l])
    else:
        l = None

    s = (torch.LongTensor([hp.unique_speakers.index(item[2])])
         if hp.multi_speaker else None)

    if torch.cuda.is_available() and not force_cpu:
        t = t.to(device)
        if l is not None:
            l = l.to(device)
        if s is not None:
            s = s.to(device)

    s = model.inference(t, speaker=s, language=l).cpu().detach().numpy()
    s = audio.denormalize_spectrogram(s, not hp.predict_linear)

    return s
