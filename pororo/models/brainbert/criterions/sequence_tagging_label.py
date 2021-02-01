# Copyright (c) Facebook, Inc. and its affiliates and Kakao Brain. All Rights Reserved

import math

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion("sequence_tagging_label")
class SequenceTaggingLabelCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.classification_head_name = args.classification_head_name

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument(
            '--classification-head-name',
            default='sequence_tagging_label_head',
            help='name of the classification head to use',
        )
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2 the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert (
            hasattr(model, "classification_heads") and
            self.classification_head_name in model.classification_heads
        ), "model must provide sentence classification head for --criterion=sequence_tagging_label"

        logits, _ = model(
            **sample["net_input"],
            **sample["segments"],
            features_only=True,
            classification_head_name=self.classification_head_name,
        )

        targets = model.get_targets(sample, [logits]).view(-1)
        logits = logits.view(-1, logits.size(-1))

        sample_size = sample["ntokens"] - sample["target"].size(
            0)  # number of tokens without eos
        loss = F.nll_loss(
            F.log_softmax(logits, dim=-1, dtype=torch.float32),
            targets,
            reduction="sum",
            ignore_index=-1,
        )

        masked_preds = logits[targets != -1].argmax(dim=1)
        masked_targets = targets[targets != -1]

        logging_output = {
            "loss": utils.item(loss.data),
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "ncorrect": utils.item((masked_preds == masked_targets).sum()),
        }

        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(
            log.get("loss", 0) for log in logging_outputs))
        ntokens = utils.item(
            sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs))
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs))

        nll_loss = loss_sum / ntokens / math.log(2)

        agg_output = {
            "loss": loss_sum / sample_size / math.log(2),
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
            "nll_loss": nll_loss,
        }

        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            accuracy = (ncorrect / sample_size) * 100
            agg_output.update(accuracy=accuracy)

        return agg_output

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
