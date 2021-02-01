# Copyright (c) Facebook, Inc. and its affiliates and Kakao Brain. All Rights Reserved

import math

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion("dependency_parse")
class DependencyParseLabelCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.classification_head_name = args.classification_head_name

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument(
            "--classification-head-name",
            default="dependency_parse_head",
            help="name of the classification head to use",
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
        ), "model must provide sentence classification head for --criterion=dependency_parse_head"

        # extract features from backbone module
        features, _ = model(
            **sample["net_input"],
            **sample["segments"],
            features_only=True,
        )
        masks = sample["net_input"]["src_tokens"] == 1

        # forward extracted features to get label and head logits
        # yapf: disable
        head_logits, label_logits = model.classification_heads[self.classification_head_name](
            features,
            masks,
        )
        # yapf: enable

        # calculate head loss
        head_targets = sample["target0"].view(-1)
        head_logits = head_logits.view(-1, head_logits.size(-1))

        head_logits = head_logits[head_targets != 0]
        head_targets = head_targets[head_targets != 0]

        head_loss = F.nll_loss(
            F.log_softmax(head_logits, dim=-1, dtype=torch.float32),
            head_targets,
            ignore_index=-1,
        )

        masked_preds = head_logits[head_targets != -1].argmax(dim=1)
        masked_targets = head_targets[head_targets != -1]

        head_ncorrect = utils.item((masked_preds == masked_targets).sum())

        sample_size = masked_targets.size(0)

        # calculate label loss
        label_targets = sample["target1"].view(-1)
        label_logits = label_logits.view(-1, label_logits.size(-1))

        label_logits = label_logits[label_targets != 0]
        label_targets = label_targets[label_targets != 0]

        label_loss = F.nll_loss(
            F.log_softmax(label_logits, dim=-1, dtype=torch.float32),
            label_targets,
            ignore_index=-1,
        )

        masked_preds = label_logits[label_targets != -1].argmax(dim=1)
        masked_targets = label_targets[label_targets != -1]

        label_ncorrect = utils.item((masked_preds == masked_targets).sum())

        loss = label_loss + head_loss

        logging_output = {
            "sample_size": sample_size,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target0"].size(0),
            "loss": utils.item(loss.data),
            "head_loss": utils.item(head_loss.data),
            "label_loss": utils.item(label_loss.data),
            "head_ncorrect": head_ncorrect,
            "label_ncorrect": label_ncorrect,
        }

        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(
            log.get("loss", 0) for log in logging_outputs))
        label_loss_sum = utils.item(
            sum(log.get("label_loss", 0) for log in logging_outputs))
        head_loss_sum = utils.item(
            sum(log.get("head_loss", 0) for log in logging_outputs))
        ntokens = utils.item(
            sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs))
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs))

        nll_loss = loss_sum / ntokens / math.log(2)
        label_loss = label_loss_sum / sample_size / math.log(2)
        head_loss = head_loss_sum / sample_size / math.log(2)

        agg_output = {
            "loss": loss_sum / sample_size / math.log(2),
            "label_loss": label_loss,
            "head_loss": head_loss,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
            "nll_loss": nll_loss,
        }

        if len(logging_outputs) > 0 and "head_ncorrect" in logging_outputs[0]:
            head_ncorrect = sum(
                log.get("head_ncorrect", 0) for log in logging_outputs)
            head_accuracy = (head_ncorrect / sample_size) * 100
            agg_output.update(head_accuracy=head_accuracy)

        if len(logging_outputs) > 0 and "label_ncorrect" in logging_outputs[0]:
            label_ncorrect = sum(
                log.get("label_ncorrect", 0) for log in logging_outputs)
            label_accuracy = (label_ncorrect / sample_size) * 100
            agg_output.update(label_accuracy=label_accuracy)

        return agg_output

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
