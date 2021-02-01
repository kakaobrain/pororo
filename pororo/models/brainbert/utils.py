# Copyright (c) Facebook, Inc., its affiliates and Kakao Brain. All Rights Reserved

import re

import numpy as np
from fairseq.data.encoders import register_bpe


@register_bpe("custom_char")
class CustomChar(object):

    def __init__(self):
        pass

    def encode(self, x: str) -> str:
        x = x.strip()

        if len(x) == 0:
            return ""

        x = [c for c in re.sub("\s+", " ", x)]

        result = list()
        for i in range(len(x)):
            if x[i] == " ":
                x[i + 1] = f"▁{x[i+1]}"
                continue
            else:
                result.append(x[i])
        result[0] = f"▁{result[0]}"
        return " ".join(result)

    def decode(self, x: str) -> str:
        return x.replace(" ", "").replace("▁", " ").strip()

    def is_beginning_of_word(self, x: str) -> bool:
        if x in ["<unk>", "<s>", "</s>", "<pad>"]:
            # special elements are always considered beginnings
            # HACK: this logic is already present in fairseq/tasks/masked_lm.py
            # but these special tokens are also contained in the sentencepiece
            # vocabulary which causes duplicate special tokens. This hack makes
            # sure that they are all taken into account.
            return True
        return x.startswith("▁")


def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return np.squeeze(y)
