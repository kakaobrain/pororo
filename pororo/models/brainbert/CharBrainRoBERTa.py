# Copyright (c) Facebook, Inc., its affiliates and Kakao Brain. All Rights Reserved

from typing import List, Union

import numpy as np
import torch
from fairseq.models.roberta import RobertaHubInterface, RobertaModel

from pororo.models.brainbert.utils import CustomChar, softmax
from pororo.tasks.utils.download_utils import download_or_load


class CharBrainRobertaModel(RobertaModel):
    """
    Helper class to load pre-trained models easily. And when you call load_hub_model,
    you can use brainbert models as same as RobertaHubInterface of fairseq.
    Methods
    -------
    load_model(log_name: str)
        Load RobertaModel. Supported names are on README.md
    load_hub_model(log_name: str)
        Load RobertaHubInterface.
    available_models
        return list of available models
    Examples::
        >>> model = BrainRobertaModel.load_hub_model('brainbert.base')
        >>> tokens = model.encode('안녕하세요.')
        >>> features = model.extract_features(tokens, return_all_hiddens=False)
        >>> features
        - tensor([[[33.0463, -5.7491, -5.6917,  ..., -7.9180, -0.7573, 17.5376],
          [ 0.4918, -1.9358, -2.0424,  ..., -4.7024, -1.7995, -1.3725],
          [ 0.7734, -1.9318, -1.3917,  ..., -0.7167, -0.4305,  3.3848],
          [ 1.9992, -1.8058, -1.7588,  ..., -1.7262, -0.7903,  0.7955]]]
    """

    @classmethod
    def load_model(cls, model_name: str, lang: str, **kwargs):
        """
        Load pre-trained model as RobertaHubInterface.
        :param model_name: model name from available_models
        :return: pre-trained model
        """
        from fairseq import hub_utils

        ckpt_dir = download_or_load(model_name, lang)
        x = hub_utils.from_pretrained(
            ckpt_dir,
            "model.pt",
            ckpt_dir,
            **kwargs,
        )

        return CharBrainRobertaHubInterface(
            x["args"],
            x["task"],
            x["models"][0],
        )


class CharBrainRobertaHubInterface(RobertaHubInterface):

    def __init__(self, args, task, model):
        super().__init__(args, task, model)
        self.bpe = CustomChar()

    def tokenize(self, sentence: str, add_special_tokens: bool = False):
        result = self.bpe.encode(sentence)
        if add_special_tokens:
            result = f"<s> {result} </s>"
        return result

    def encode(
        self,
        sentence: str,
        *addl_sentences,
        add_special_tokens: bool = True,
        no_separator: bool = False,
    ) -> torch.LongTensor:
        """
        BPE-encode a sentence (or multiple sentences).
        Every sequence begins with a beginning-of-sentence (`<s>`) symbol.
        Every sentence ends with an end-of-sentence (`</s>`) and we use an
        extra end-of-sentence (`</s>`) as a separator.
        """
        bpe_sentence = self.tokenize(
            sentence,
            add_special_tokens=add_special_tokens,
        )

        for s in addl_sentences:
            bpe_sentence += " </s>" if not no_separator and add_special_tokens else ""
            bpe_sentence += (" " + self.tokenize(
                s,
                add_special_tokens=False,
            ) + " </s>" if add_special_tokens else "")

        tokens = self.task.source_dictionary.encode_line(
            bpe_sentence,
            append_eos=False,
            add_if_not_exist=False,
        )
        return tokens.long()

    def decode(
        self,
        tokens: torch.LongTensor,
        skip_special_tokens: bool = True,
        remove_bpe: bool = True,
    ) -> str:
        assert tokens.dim() == 1
        tokens = tokens.numpy()

        if tokens[0] == self.task.source_dictionary.bos(
        ) and skip_special_tokens:
            tokens = tokens[1:]  # remove <s>

        eos_mask = tokens == self.task.source_dictionary.eos()
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)

        if skip_special_tokens:
            sentences = [
                np.array(
                    [c
                     for c in s
                     if c != self.task.source_dictionary.eos()])
                for s in sentences
            ]

        sentences = [
            " ".join([self.task.source_dictionary.symbols[c]
                      for c in s])
            for s in sentences
        ]
        if remove_bpe:
            sentences = [self.bpe.decode(s) for s in sentences]
        if len(sentences) == 1:
            return sentences[0]
        return sentences

    @torch.no_grad()
    def predict_output(
        self,
        sentence: str,
        *addl_sentences,
        add_special_tokens: bool = True,
        no_separator: bool = False,
        show_probs: bool = False,
    ) -> Union[str, float]:
        """Predict output, either a classification label or regression target,
         using a fine-tuned sentence prediction model.
        :returns output
            str (classification) or float (regression)
            >>> from brain_bert import BrainRobertaModel
            >>> model = BrainRobertaModel.load_model('brainbert.base.ko.kornli')
            >>> model.predict_output(
            ...    'BrainBert는 한국어 코퍼스에 학습된 언어모델이다.',
            ...    'BrainBert는 한국어 모델이다.',
            ...    )
            entailment
            >>> model = BrainRobertaModel.load_model('brainbert.base.ko.korsts')
            >>> model.predict_output(
            ...    'BrainBert는 한국어 코퍼스에 학습된 언어모델이다.',
            ...    'BrainBert는 한국어 모델이다.',
            ...    )
            0.8374465107917786
        """
        assert self.args.task == "sentence_prediction", (
            "predict_output() only works for sentence prediction tasks.\n"
            "Use predict() to obtain model outputs; "
            "use predict_span() for span prediction tasks.")
        assert (
            "sentence_classification_head" in self.model.classification_heads
        ), "need pre-trained sentence_classification_head to make predictions"

        tokens = self.encode(
            sentence,
            *addl_sentences,
            add_special_tokens=add_special_tokens,
            no_separator=no_separator,
        )
        regression_target = self.args.regression_target
        with torch.no_grad():
            prediction = self.predict(
                "sentence_classification_head",
                tokens,
                return_logits=regression_target,
            )
            if regression_target:
                prediction = prediction.item()  # float
            else:
                label_fn = lambda label: self.task.label_dictionary.string(
                    [label + self.task.label_dictionary.nspecial])

                if show_probs:
                    probs = softmax(prediction.cpu().numpy())
                    probs = probs.tolist()

                    res = {label_fn(i): prob for i, prob in enumerate(probs)}
                    return res

                prediction = label_fn(prediction.argmax().item())  # str

        return prediction

    @torch.no_grad()
    def predict_tags(
        self,
        sentence: Union[List[str], str],
        add_special_tokens: bool = True,
        no_separator: bool = False,
    ):
        label_fn = lambda label: self.task.label_dictionary.string([label])

        if isinstance(sentence, list):
            lengths = [len(self.tokenize(sent).split()) for sent in sentence]
            max_len = max(lengths)
            res_sentence = []
            for sent in sentence:
                tokens = self.encode(
                    sent,
                    add_special_tokens=add_special_tokens,
                    no_separator=no_separator,
                )
                to_pad = max_len - (len(tokens) - 2)
                pad_tokens = [self.task.source_dictionary.index("<pad>")
                             ] * to_pad
                if pad_tokens:
                    tokens = torch.cat([tokens, torch.tensor(pad_tokens)])
                res_sentence.append(tokens)
            li_sentence = torch.stack(res_sentence)

            results = (self.predict(
                "sequence_tagging_head",
                li_sentence,
            )[:, 1:-1, :].argmax(dim=-1).cpu().numpy())
            labels = [[
                label_fn(int(pred) + self.task.label_dictionary.nspecial)
                for pred in preds
            ]
                      for preds in results]

            return [[
                (token, label)
                for token, label in zip(self.tokenize(sent).split(), label)
            ]
                    for sent, label in zip(sentence, labels)]

        tokens = self.encode(
            sentence,
            add_special_tokens=add_special_tokens,
            no_separator=no_separator,
        )

        # Get first batch and ignore <s> & </s> tokens
        preds = (self.predict(
            "sequence_tagging_head",
            tokens,
        )[0, 1:-1, :].argmax(dim=1).cpu().numpy())
        labels = [
            label_fn(int(pred) + self.task.label_dictionary.nspecial)
            for pred in preds
        ]

        return [
            (token, label)
            for token, label in zip(self.tokenize(sentence).split(), labels)
        ]
