# Copyright (c) Facebook, Inc., its affiliates and Kakao Brain. All Rights Reserved

from typing import Union

import numpy as np
import torch
from fairseq.models.roberta import RobertaHubInterface, RobertaModel

from pororo.models.brainbert.utils import softmax
from pororo.tasks.utils.download_utils import download_or_load
from pororo.tasks.utils.tokenizer import CustomTokenizer


class BrainRobertaModel(RobertaModel):
    """
    Helper class to load pre-trained models easily. And when you call load_hub_model,
    you can use brainbert models as same as RobertaHubInterface of fairseq.
    Methods
    -------
    load_model(log_name: str): Load RobertaModel

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
        tok_path = download_or_load(f"tokenizers/bpe32k.{lang}.zip", lang)

        x = hub_utils.from_pretrained(
            ckpt_dir,
            "model.pt",
            ckpt_dir,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return BrainRobertaHubInterface(
            x["args"],
            x["task"],
            x["models"][0],
            tok_path,
        )


class BrainRobertaHubInterface(RobertaHubInterface):

    def __init__(self, args, task, model, tok_path):
        super().__init__(args, task, model)
        self.bpe = CustomTokenizer.from_file(
            vocab_filename=f"{tok_path}/vocab.json",
            merges_filename=f"{tok_path}/merges.txt",
        )

    def tokenize(self, sentence: str, add_special_tokens: bool = False):
        result = " ".join(self.bpe.encode(sentence).tokens)
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
        Example (single sentence): `<s> a b c </s>`
        Example (sentence pair): `<s> d e f </s> </s> 1 2 3 </s>`
        The BPE encoding follows GPT-2. One subtle detail is that the GPT-2 BPE
        requires leading spaces. For example::
            >>> roberta.encode('Hello world').tolist()
            [0, 31414, 232, 2]
            >>> roberta.encode(' world').tolist()
            [0, 232, 2]
            >>> roberta.encode('world').tolist()
            [0, 8331, 2]
        """
        bpe_sentence = self.tokenize(
            sentence,
            add_special_tokens=add_special_tokens,
        )

        for s in addl_sentences:
            bpe_sentence += " </s>" if not no_separator and add_special_tokens else ""
            bpe_sentence += (" " + self.tokenize(s, add_special_tokens=False) +
                             " </s>" if add_special_tokens else "")
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
            sentences = [
                s.replace(" ", "").replace("▁", " ").strip() for s in sentences
            ]
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

        with torch.no_grad():
            prediction = self.predict(
                "sentence_classification_head",
                tokens,
                return_logits=self.args.regression_target,
            )
            if self.args.regression_target:
                return prediction.item()  # float

            label_fn = lambda label: self.task.label_dictionary.string(
                [label + self.task.label_dictionary.nspecial])

            if show_probs:
                probs = softmax(prediction.cpu().numpy())
                probs = probs.tolist()
                probs = {label_fn(i): prob for i, prob in enumerate(probs)}
                return probs

        return label_fn(prediction.argmax().item())  # str

    @torch.no_grad()
    def predict_span(
        self,
        question: str,
        context: str,
        add_special_tokens: bool = True,
        no_separator: bool = False,
    ) -> str:
        """
        Predict span from context using a fine-tuned span prediction model.

        :returns answer
            str

        >>> from brain_bert import BrainRobertaModel
        >>> model = BrainRobertaModel.load_model('brainbert.base.ko.korquad')
        >>> model.predict_span(
        ...    'BrainBert는 어떤 언어를 배운 모델인가?',
        ...    'BrainBert는 한국어 코퍼스에 학습된 언어모델이다.',
        ...    )
        한국어

        """
        assert self.args.task == "span_prediction", (
            "predict_span() only works for span prediction tasks.\n"
            "Use predict() to obtain model outputs (e.g., logits); "
            "use predict_output() for sentence prediction tasks.")

        max_length = self.task.max_positions()
        tokens = self.encode(
            question,
            context,
            add_special_tokens=add_special_tokens,
            no_separator=no_separator,
        )[:max_length]
        with torch.no_grad():
            logits = self.predict(
                "span_prediction_head",
                tokens,
                return_logits=True,
            ).squeeze()  # T x 2
            # first predict start position,
            # then predict end position among the remaining logits
            start = logits[:, 0].argmax().item()
            mask = (torch.arange(
                logits.size(0), dtype=torch.long, device=self.device) >= start)
            end = (mask * logits[:, 1]).argmax().item()
            # end position is shifted during training, so we add 1 back
            answer_tokens = tokens[start:end + 1]

            answer = ""
            if len(answer_tokens) >= 1:
                decoded = self.decode(answer_tokens)
                if isinstance(decoded, str):
                    answer = decoded

        return (answer, (start, end + 1))

    @torch.no_grad()
    def predict_tags(
        self,
        sentence: str,
        add_special_tokens: bool = True,
        no_separator: bool = False,
    ):
        tokens = self.encode(
            sentence,
            add_special_tokens=add_special_tokens,
            no_separator=no_separator,
        )

        label_fn = lambda label: self.task.label_dictionary.string([label])

        # Get first batch and ignore <s> & </s> tokens
        preds = (self.predict(
            "sequence_tagging_head",
            tokens,
        )[0, 1:-1, :].argmax(dim=1).cpu().numpy())
        labels = [
            label_fn(int(pred) + self.task.label_dictionary.nspecial)
            for pred in preds
        ]
        return [(
            token,
            label,
        ) for token, label in zip(self.tokenize(sentence).split(), labels)]
