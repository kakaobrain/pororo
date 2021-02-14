import re

import regex

try:
    import epitran  # noqa
except ImportError:
    raise ImportError("Please install epitran with: `pip install epitran`")
try:
    from ko_pron import romanise
except ImportError:
    raise ImportError("Please install ko_pron with: `pip install ko_pron`")

from pororo.models.tts.tacotron.params import Params as hp

_pad = "_"  # a dummy character for padding sequences to align text in batches to the same length
_eos = "~"  # character which marks the end of a sequnce, further characters are invalid
_unk = "@"  # symbols which are not in hp.characters and are present are substituted by this


def jejueo_romanize(text):
    word = ""
    results = []
    for char in text:
        if regex.search("\p{Hangul}", char) is not None:
            word += char
        else:
            result = romanise(word, "rr")
            results.append(result)
            word = char
    result = romanise(word, "rr")
    results.append(result)
    return "".join(results)


def _other_symbols():
    return [_pad, _eos, _unk] + list(hp.punctuations_in) + list(
        hp.punctuations_out)


def to_lower(text):
    """Convert uppercase text into lowercase."""
    return text.lower()


def remove_odd_whitespaces(text):
    """Remove multiple and trailing/leading whitespaces."""
    return " ".join(text.split())


def remove_punctuation(text):
    """Remove punctuation from text."""
    punct_re = "[" + hp.punctuations_out + hp.punctuations_in + "]"
    return re.sub(punct_re.replace("-", "\-"), "", text)


def to_sequence(text, use_phonemes=False):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text."""
    transform_dict = {
        s: i for i, s in
        enumerate(_other_symbols() +
                  list(hp.phonemes if use_phonemes else hp.characters))
    }
    sequence = [
        transform_dict[_unk] if c not in transform_dict else transform_dict[c]
        for c in text
    ]
    sequence.append(transform_dict[_eos])
    return sequence


def to_text(sequence, use_phonemes=False):
    """Converts a sequence of IDs back to a string"""
    transform_dict = {
        i: s for i, s in
        enumerate(_other_symbols() +
                  list(hp.phonemes if use_phonemes else hp.characters))
    }
    result = ""
    for symbol_id in sequence:
        if symbol_id in transform_dict:
            s = transform_dict[symbol_id]
            if s == _eos:
                break
            result += s
    return result


def romanize(text):
    """
    Copied from https://github.com/kord123/ko_pron
    Copyright (c) Andriy Koretskyy
    """
    word = ""
    results = []
    for char in text:
        if regex.search("\p{Hangul}", char) is not None or char == " ":
            word += char
        elif char.isalpha():
            result = romanise(word, "rr")
            results.append(result)
            word = char
    result = romanise(word, "rr")
    results.append(result)
    return "".join(results)
