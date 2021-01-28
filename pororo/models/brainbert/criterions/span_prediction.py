# Copyright (c) Facebook, Inc. and its affiliates and Kakao Brain. All Rights Reserved

import math

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion("span_prediction")
class SpanPredictionCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        if self.args.save_predictions is not None:
            self.prediction_h = open(self.args.save_predictions, "w")
        else:
            self.prediction_h = None

    def __del__(self):
        if self.prediction_h is not None:
            self.prediction_h.close()

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument(
            '--save-predictions',
            metavar='FILE',
            help='file to save predictions to',
        )
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute span loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert (
            hasattr(model, "classification_heads") and
            "span_prediction_head" in model.classification_heads
        ), "model must provide span prediction head for --criterion=span_prediction"

        # logits: (sample_size, src_length, 2)
        logits, _ = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name="span_prediction_head",
        )
        sample_size = logits.size(0)

        if "target" in sample:
            # targets: (sample_size, 2)
            targets = model.get_targets(sample, [logits])
            assert targets.max() < logits.size(
                1), f"{targets.max()}, {logits.size()}, {targets.size()}"
            assert logits.size(0) == targets.size(
                0), f"{targets.size()}, {logits.size()}"
            assert (logits.size(2) == targets.size(1) ==
                    2), f"{targets.size()}, {logits.size()}"
            # loss = start_loss + end_loss (also summed over samples)
            # TODO: mask over padded positions?
            loss = F.nll_loss(
                F.log_softmax(logits, dim=1, dtype=torch.float32).transpose(
                    1, 2).reshape(-1, logits.size(1)),
                targets.reshape(-1),
                reduction="sum",
            )
        else:
            targets = None
            loss = torch.tensor(0.0, requires_grad=True)

        # {id}\t{pred_start} {pred_end}\t{target_start} {target_end}
        # - make sure that end is never before start
        # - save after subtracting question length
        if self.prediction_h is not None:
            preds_start = logits[:, :, 0].argmax(dim=1)  # N
            mask = (torch.arange(logits.size(1)).to(preds_start) >=
                    preds_start[..., None])  # N x T
            preds_end = (mask * logits[:, :, 1]).argmax(dim=1)

            question_lengths = sample["net_input"]["input0_lengths"]
            offsets = question_lengths + (self.args.separator_token is not None)

            for i, (id, pred_start, pred_end, offset) in enumerate(
                    zip(
                        sample["id"].tolist(),
                        preds_start.tolist(),
                        preds_end.tolist(),
                        offsets.tolist(),
                    )):
                if targets is not None:
                    label = targets[i].tolist()
                    print(
                        "{}\t{} {}\t{} {}".format(
                            id,
                            pred_start - offset,
                            pred_end - offset,
                            label[0] - offset,
                            label[1] - offset,
                        ),
                        file=self.prediction_h,
                    )
                else:
                    print(
                        "{}\t{} {}".format(
                            id,
                            pred_start - offset,
                            pred_end - offset,
                        ),
                        file=self.prediction_h,
                    )

        logging_output = {
            "loss": utils.item(loss.data) if reduce else loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample_size,
            "sample_size": sample_size,
        }
        if targets is not None:
            _, preds = logits.max(dim=1)
            ncorrect_start = (preds[:, 0] == targets[:, 0]).sum().item()
            ncorrect_end = (preds[:, 1] == targets[:, 1]).sum().item()
            ncorrect_span = (preds == targets).min(dim=1)[0].sum().item()
            logging_output.update(
                ncorrect_start=ncorrect_start,
                ncorrect_end=ncorrect_end,
                ncorrect_span=ncorrect_span,
            )
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        agg_output = {
            "loss": loss_sum / sample_size / math.log(2),
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

        if len(logging_outputs) > 0 and "ncorrect_span" in logging_outputs[0]:
            ncorrect_start = sum(
                log.get("ncorrect_start", 0) for log in logging_outputs)
            ncorrect_end = sum(
                log.get("ncorrect_end", 0) for log in logging_outputs)
            ncorrect_span = sum(
                log.get("ncorrect_span", 0) for log in logging_outputs)
            agg_output.update(accuracy_start=ncorrect_start / nsentences)
            agg_output.update(accuracy_end=ncorrect_end / nsentences)
            agg_output.update(accuracy_span=ncorrect_span /
                              nsentences)  # approximate Exact Match (EM)

        if sample_size != ntokens:
            agg_output["nll_loss"] = loss_sum / ntokens / math.log(2)
        return agg_output
