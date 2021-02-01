# Copyright (c) Facebook, Inc., its affiliates and Kakao Brain. All Rights Reserved

from typing import Union

import torch
from fairseq.models.roberta import RobertaModel
from fairseq.models.roberta.hub_interface import RobertaHubInterface
from transformers import BertJapaneseTokenizer

from pororo.models.brainbert.utils import softmax
from pororo.tasks.utils.download_utils import download_or_load


class JabertaModel(RobertaModel):

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
            load_checkpoint_heads=True,
            **kwargs,
        )
        return JabertaHubInterface(x["args"], x["task"], x["models"][0])


class JabertaHubInterface(RobertaHubInterface):

    def __init__(self, args, task, model):
        super().__init__(args, task, model)
        try:
            import ipadic  # noqa
        except ImportError:
            raise ImportError(
                "Please install ipadic with: `pip install ipadic`")
        try:
            import fugashi  # noqa
        except ImportError:
            raise ImportError(
                "Please install fugashi with: `pip install fugashi`")
        self.bpe = BertJapaneseTokenizer.from_pretrained(
            "cl-tohoku/bert-base-japanese-whole-word-masking")

    def tokenize(self, sentence: str, add_special_tokens: bool = False):
        result = " ".join(self.bpe.tokenize(sentence)[:510])
        if add_special_tokens:
            result = f"<s> {result} </s>"
        return result

    def encode(
        self,
        sentence: str,
        *addl_sentences,
        add_special_tokens: bool = True,
        no_separator: bool = False,
        return_bpe: bool = False,
    ) -> torch.LongTensor:
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
        if return_bpe:
            return tokens.long(), bpe_sentence.split()[1:-1]
        return tokens.long()

    def fill_mask(self, masked_input: str, topk: int = 5):
        masked_token = "__"
        assert (
            masked_token in masked_input and
            masked_input.count(masked_token) == 1
        ), "Please add one {0} token for the input, eg: 'He is a {0} guy'".format(
            masked_token)

        text_spans = masked_input.split(masked_token)
        text_spans_bpe = ((" {0} ".format("<mask>")).join([
            " ".join(self.bpe.tokenize(text_span.rstrip()))
            for text_span in text_spans
        ]).strip())
        tokens = self.task.source_dictionary.encode_line(
            "<s> " + text_spans_bpe + " </s>",
            append_eos=False,
            add_if_not_exist=False,
        )

        masked_index = torch.nonzero(
            tokens == self.task.mask_idx,
            as_tuple=False,
        )

        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        with torch.no_grad():
            features, _ = self.model(
                tokens.long().to(device=self.device),
                features_only=False,
                return_all_hiddens=False,
            )
        logits = features[0, masked_index, :].squeeze()
        prob = logits.softmax(dim=0)
        _, index = prob.topk(k=topk, dim=0)
        topk_predicted_token_bpe = self.task.source_dictionary.string(index)
        return [
            bpe.replace("##", "") for bpe in topk_predicted_token_bpe.split()
        ]

    @torch.no_grad()
    def predict_output(
        self,
        sentence: str,
        *addl_sentences,
        add_special_tokens: bool = True,
        no_separator: bool = False,
        show_probs: bool = False,
    ) -> Union[str, float]:
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
    def predict_tags(self, sentence: str, no_separator: bool = False):
        label_fn = lambda label: self.task.label_dictionary.string([label])
        tokens, words = self.encode(
            sentence,
            no_separator=no_separator,
            return_bpe=True,
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

        return [(word, label) for word, label in zip(words, labels)]
