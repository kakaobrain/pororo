# Copyright (c) Facebook, Inc., its affiliates and Kakao Brain. All Rights Reserved

from typing import Union

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models.roberta import RobertaModel
from fairseq.models.roberta.hub_interface import RobertaHubInterface

from pororo.models.brainbert.utils import softmax
from pororo.tasks.utils.download_utils import download_or_load


class CustomRobertaModel(RobertaModel):

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
            bpe="gpt2",
            load_checkpoint_heads=True,
            **kwargs,
        )
        return CustomRobertaHubInterface(x["args"], x["task"], x["models"][0])


class CustomRobertaHubInterface(RobertaHubInterface):

    def __init__(self, args, task, model):
        args.gpt2_encoder_json = download_or_load("misc/encoder.json", "en")
        args.gpt2_vocab_bpe = download_or_load("misc/vocab.bpe", "en")
        super().__init__(args, task, model)
        self.softmax = nn.Softmax(dim=1)

    @torch.no_grad()
    def predict_output(
        self,
        sentence: str,
        *addl_sentences,
        no_separator: bool = False,
        show_probs: bool = False,
    ) -> Union[str, float]:
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

    def fill_mask(self, masked_input: str, topk: int = 5):
        masked_token = "<mask>"
        masked_input = masked_input.replace("__", masked_token)

        assert (
            masked_token in masked_input and
            masked_input.count(masked_token) == 1
        ), "Please add one {0} token for the input, eg: 'He is a {0} guy'".format(
            masked_token)

        text_spans = masked_input.split(masked_token)
        text_spans_bpe = ((" {0} ".format(masked_token)).join([
            self.bpe.encode(text_span.rstrip()) for text_span in text_spans
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

        with utils.model_eval(self.model):
            features, _ = self.model(
                tokens.long().to(device=self.device),
                features_only=False,
                return_all_hiddens=False,
            )

        logits = features[0, masked_index, :].squeeze()
        prob = logits.softmax(dim=0)
        _, index = prob.topk(k=topk, dim=0)
        topk_predicted_token_bpe = self.task.source_dictionary.string(index)

        topk_filled_outputs = []
        for index, predicted_token_bpe in enumerate(
                topk_predicted_token_bpe.split(" ")):
            predicted_token = self.bpe.decode(predicted_token_bpe)
            # Quick hack to fix https://github.com/pytorch/fairseq/issues/1306
            if predicted_token_bpe.startswith("\u2581"):
                predicted_token = " " + predicted_token
            if " {0}".format(masked_token) in masked_input:
                topk_filled_outputs.append(predicted_token)
            else:
                topk_filled_outputs.append(predicted_token)
        return topk_filled_outputs

    @torch.no_grad()
    def predict_segments(self, sentence: str, no_separator: bool = False):
        label_fn = lambda label: self.task.label_dictionary.string([label])
        tokens = self.encode(sentence, no_separator=no_separator)
        preds = self.predict("sequence_tagging_head", tokens)[0, 1:-1, :]
        probs = self.softmax(preds).cpu().numpy()

        res_prob = list()
        for prob in probs:
            prob = {
                label_fn(i + self.task.label_dictionary.nspecial):
                round(p * 100, 2) for i, p in enumerate(prob.tolist())
            }
            res_prob.append(prob)

        preds = preds.argmax(dim=1).cpu().numpy()
        labels = [
            label_fn(int(pred) + self.task.label_dictionary.nspecial)
            for pred in preds
        ]

        # Set first subword's prediction as a token prediction
        n_tokens = 0
        token_preds = [0]
        for idx, token in enumerate(sentence.split()):
            if idx != 0:
                token_preds.append(n_tokens)
                token = f" {token}"
            n_tokens += len(self.bpe.encode(token).split())

        return [(token, labels[lab], res_prob[lab])
                for token, lab in zip(sentence.split(), token_preds)]

    @torch.no_grad()
    def predict_tags(self, sentence: str, no_separator: bool = False):
        label_fn = lambda label: self.task.label_dictionary.string([label])

        tokens = self.encode(sentence, no_separator=no_separator)

        # Get first batch and ignore <s> & </s> tokens
        preds = (self.predict(
            "sequence_tagging_head",
            tokens,
        )[0, 1:-1, :].argmax(dim=1).cpu().numpy())
        labels = [
            label_fn(int(pred) + self.task.label_dictionary.nspecial)
            for pred in preds
        ]

        return [(self.decode(token.unsqueeze(0)), label)
                for token, label in zip(tokens[1:-1], labels)]
