# Copyright (c) UKP Lab, its affiliates and Kakao Brain. All Rights Reserved

import json
import logging
import os
from typing import List

from torch import nn
from transformers import PreTrainedTokenizer, RobertaModel

from pororo.tasks.utils.download_utils import download_or_load
from pororo.tasks.utils.tokenizer import CustomTokenizer


class BrainRoBERTa(nn.Module):
    """Brain RoBERTa model to generate token embeddings.

    Each token is mapped to an output vector from Brain RoBERTa.
    """

    def __init__(self, model_path: str, max_seq_length: int = 512):
        super(BrainRoBERTa, self).__init__()
        self.config_keys = ["max_seq_length"]

        if max_seq_length > 511:
            logging.warning(
                "RoBERTa only allows a max_seq_length of 511 (514 with special tokens). Value will be set to 511"
            )
            max_seq_length = 511
        self.max_seq_length = max_seq_length

        lang = "ko"
        dict_path = download_or_load("misc/spm.ko.vocab", lang)
        tok_path = download_or_load(
            "tokenizers/bpe32k.ko.zip",
            lang,
        )

        self.roberta = RobertaModel.from_pretrained(model_path)
        self.tokenizer = BrainRobertaTokenizer(dict_path, tok_path)

        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.mask_token_id = self.tokenizer.mask_token_id

    def forward(self, features):
        output_states = self.roberta(**features)
        output_tokens = output_states[0]

        cls_tokens = output_tokens[:, 0, :]  # CLS token is first token
        features.update({
            "token_embeddings": output_tokens,
            "cls_token_embeddings": cls_tokens,
            "attention_mask": features["attention_mask"],
        })

        if self.roberta.config.output_hidden_states:
            all_layer_idx = 2
            # Some models only output last_hidden_states and all_hidden_states
            if (len(output_states) < 3):
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({"all_layer_embeddings": hidden_states})

        return features

    def get_word_embedding_dimension(self) -> int:
        return self.roberta.config.hidden_size

    def tokenize(self, text: List[str]):
        output = {}

        t = [text]

        output.update(
            self.tokenizer(
                *t,
                padding=True,
                truncation="longest_first",
                return_tensors="pt",
                max_length=self.max_seq_length,
            ))
        return output

    def get_sentence_features(self, tokens: List[int], pad_seq_length: int):
        pad_seq_length = min(pad_seq_length, self.max_seq_length) + 3

        # yapf: disable
        tokens = ([self.cls_token_id] + tokens + [self.sep_token_id] + [self.sep_token_id])
        # yapf: enable
        return self.tokenizer.prepare_for_model(
            tokens,
            max_length=pad_seq_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
            prepend_batch_axis=True,
        )

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        self.roberta.save_pretrained(output_path)

        with open(
                os.path.join(output_path, "sentence_brain_roberta_config.json"),
                "w",
        ) as fout:
            json.dump(self.get_config_dict(), fout, indent=2)

    @staticmethod
    def load(input_path: str):
        with open(
                os.path.join(input_path, "sentence_brain_roberta_config.json"),
                "r",
        ) as fin:
            config = json.load(fin)
        return BrainRoBERTa(input_path, **config)


class BrainRobertaTokenizer(PreTrainedTokenizer):

    def __init__(
        self,
        vocab_file,
        tokenizer_path,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        **kwargs,
    ):
        super().__init__()
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.mask_token = mask_token

        self.tokenizer = CustomTokenizer.from_file(
            vocab_filename=f"{tokenizer_path}/vocab.json",
            merges_filename=f"{tokenizer_path}/merges.txt",
        )

        brain_tokens = [bos_token, pad_token, eos_token, unk_token]

        with open(vocab_file, "r") as r:
            brain_tokens.extend(
                [line.strip().split()[0] for line in r.readlines()][3:])
        brain_tokens.append(mask_token)

        self.brain_tok2idx = {tok: idx for idx, tok in enumerate(brain_tokens)}
        self.brain_idx2tok = {idx: tok for idx, tok in enumerate(brain_tokens)}

    @property
    def vocab_size(self):
        return len(self.brain_idx2tok)

    def _convert_token_to_id(self, token):
        return self.brain_tok2idx.get(
            token,
            self.brain_tok2idx.get(self.unk_token),
        )

    def _convert_id_to_token(self, index):
        return self.brain_idx2tok.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        return "".join(tokens).replace("▁", " ").strip()

    def tokenize(self, text):
        return self.tokenizer.encode(text).tokens

    def convert_tokens_to_ids(self, tokens):
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)

        tokens = [self.bos_token] + tokens + [self.eos_token] + [self.eos_token]
        return [self._convert_token_to_id(token) for token in tokens]

    def bpe(self, sent):
        return " ".join(self.tokenizer.encode(sent).tokens).strip()
