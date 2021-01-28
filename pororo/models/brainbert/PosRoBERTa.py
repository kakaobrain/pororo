# Copyright (c) Facebook, Inc., its affiliates and Kakao Brain. All Rights Reserved

import torch
from fairseq.models.roberta import RobertaHubInterface, RobertaModel

from pororo.tasks.tokenization import PororoTokenizationFactory
from pororo.tasks.utils.download_utils import download_or_load


class PosRobertaModel(RobertaModel):
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

        # cache directory is treated as the home directory for both model and data files
        ckpt_dir = download_or_load(model_name, lang)
        x = hub_utils.from_pretrained(
            ckpt_dir,
            "model.pt",
            ckpt_dir,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return PosRobertaHubInterface(
            x["args"],
            x["task"],
            x["models"][0],
            lang,
        )


class PosRobertaHubInterface(RobertaHubInterface):

    def __init__(self, args, task, model, lang):
        super().__init__(args, task, model)
        self.bpe = PororoTokenizationFactory(
            task="tokenization",
            lang="ko",
            model="mecab_ko",
        )
        self.bpe = self.bpe.load("cuda")

    def tokenize(self, sentence: str, add_special_tokens: bool = False):
        result = " ".join([token for token in self.bpe(sentence)])
        return f"<s> {result} </s>" if add_special_tokens else result

    def fill_mask(self, masked_input: str, topk: int = 15):
        mask = "__"
        assert (mask in masked_input and masked_input.count(mask)
                == 1), "Please add one {0} token for the input".format(mask)

        text_spans = masked_input.split(mask)
        text_spans_bpe = ((" {0} ".format("<mask>")).join([
            " ".join([
                token if token != " " else "▃"
                for token in self.bpe(text_span.rstrip())
            ])
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
        return [bpe for bpe in topk_predicted_token_bpe.split()]

    def encode(
        self,
        sentence: str,
        *addl_sentences,
        add_special_tokens: bool = True,
        no_separator: bool = False,
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

        return tokens.long()
