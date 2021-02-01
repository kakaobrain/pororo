# Copyright (c) Facebook, Inc. and its affiliates and Kakao Brain. All Rights Reserved

import logging
import os

import numpy as np
from fairseq import utils
from fairseq.data import (
    ConcatSentencesDataset,
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PrependTokenDataset,
    RawLabelDataset,
    RightPadDataset,
    SortDataset,
    TruncateDataset,
    data_utils,
)
from fairseq.tasks import FairseqTask, register_task
from torch import nn

logger = logging.getLogger(__name__)


@register_task("span_prediction")
class SpanPredictionTask(FairseqTask):
    """
    Span prediction task given context and question (e.g., SQuAD).
    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", metavar="FILE", help="file prefix for data")
        parser.add_argument(
            "--num-classes",
            type=int,
            help="number of sentences to be ranked",
        )
        parser.add_argument(
            "--init-token",
            type=int,
            help="add token at the beginning of each batch item",
        )
        parser.add_argument(
            "--separator-token",
            type=int,
            help="add separator token between inputs",
        )
        parser.add_argument(
            "--no-shuffle",
            action="store_true",
        )
        parser.add_argument(
            "--truncate-sequence",
            action="store_true",
            help="Truncate sequence to max_positions",
        )
        parser.add_argument(
            "--max-context-length",
            type=int,
            help="max length for each context (input1)",
        )

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary

    @classmethod
    def load_dictionary(cls, filename):
        """
        Load the dictionary from the filename

        Args:
            filename (str): the filename

        """
        dictionary = Dictionary.load(filename)
        dictionary.add_symbol("<mask>")
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        # load data dictionary
        data_dict = cls.load_dictionary(
            os.path.join(args.data, "input0", "dict.txt"),)
        logger.info("[input] dictionary: {} types".format(len(data_dict)))
        return SpanPredictionTask(args, data_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        def get_path(type, split):
            return os.path.join(self.args.data, type, split)

        def make_dataset(type, dictionary):
            split_path = get_path(type, split)

            dataset = data_utils.load_indexed_dataset(
                split_path,
                self.source_dictionary,
                self.args.dataset_impl,
                combine=combine,
            )
            return dataset

        # inputs are loaded similarly to sentence_prediction
        input0 = make_dataset("input0", self.source_dictionary)  # question
        input1 = make_dataset("input1", self.source_dictionary)  # context

        # src_tokens: <init_token> input0 <separator_token> input1 <eos_token>
        if self.args.init_token is not None:
            input0 = PrependTokenDataset(input0, self.args.init_token)
        if self.args.separator_token is not None:
            input1 = PrependTokenDataset(input1, self.args.separator_token)
        if self.args.max_context_length is not None:
            input1 = TruncateDataset(input1, self.args.max_option_length)
        src_tokens = ConcatSentencesDataset(input0, input1)
        if self.args.truncate_sequence:
            src_tokens = TruncateDataset(src_tokens, self.args.max_positions)

        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(len(src_tokens))

        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens":
                    RightPadDataset(
                        src_tokens,
                        pad_idx=self.source_dictionary.pad(),
                    ),
                "src_lengths":
                    NumelDataset(src_tokens, reduce=False),
                "input0_lengths":
                    NumelDataset(
                        input0, reduce=False
                    ),  # question length (init_token possibly included)
            },
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens, reduce=True),
        }

        # labels (spans) are loaded similarly to sentence_ranking
        label_path = "{}.label".format(get_path("label", split))

        def _process_label(positions, input0_length, truncate_sequence,
                           max_positions):
            """Process a span [start:end] to the input range.
            After processing, tokens can be accessed by tokens[start:end+1].
            TODO: change inputs to reflect this change in the first place.
            """
            start, end = [
                pos + input0_length + (self.args.separator_token is not None)
                for pos in positions
            ]
            end -= 1  # [0, 511]
            if truncate_sequence:
                if start >= max_positions:
                    start, end = max_positions - 1, max_positions - 1  # not predictable
                elif end >= max_positions:
                    end = max_positions - 1
            return start, end

        if os.path.exists(label_path):
            with open(label_path) as h:
                dataset.update(target=RawLabelDataset([
                    _process_label(
                        tuple(int(pos) for pos in x.split()),
                        dataset["net_input"]["input0_lengths"][i],
                        self.args.truncate_sequence,
                        self.max_positions(),
                    ) for i, x in enumerate(
                        h.readlines())  # (start_position, end_position)
                ]))

        nested_dataset = NestedDictionaryDataset(
            dataset,
            sizes=[src_tokens.sizes],
        )

        if self.args.no_shuffle:
            dataset = nested_dataset
        else:
            dataset = SortDataset(
                nested_dataset,
                # shuffle
                sort_order=[shuffle],
            )

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, args):
        from fairseq import models

        model = models.build_model(args, self)

        # register a SpanPredictionHead
        register_span_prediction_head(
            model,
            args,
            "span_prediction_head",
            num_classes=2,
        )
        assert "span_prediction_head" in model.classification_heads
        return model

    def max_positions(self):
        return self.args.max_positions

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary


"""
The following code should be included in the model definition,
but we keep them here to allow external fairseq installations.
"""


class SpanPredictionHead(nn.Module):
    """Head for span prediction tasks.
    Can be viewed as a 2-class output layer that is applied to every position.
    """

    def __init__(self, input_dim, inner_dim, num_classes, activation_fn,
                 pooler_dropout):
        assert num_classes == 2
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features  # take features across ALL positions
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x  # B x T x C, but softmax should be taken over T


def register_span_prediction_head(
    model,
    args,
    name,
    num_classes=None,
    inner_dim=None,
    **kwargs,
):
    """Register a span prediction head for a RobertaModel."""
    if name in model.classification_heads:
        prev_num_classes = model.classification_heads[
            name].out_proj.out_features
        prev_inner_dim = model.classification_heads[name].dense.out_features
        if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
            print(
                'WARNING: re-registering head "{}" with num_classes {} (prev: {}) '
                "and inner_dim {} (prev: {})".format(
                    name,
                    num_classes,
                    prev_num_classes,
                    inner_dim,
                    prev_inner_dim,
                ))
    model.classification_heads[name] = SpanPredictionHead(
        args.encoder_embed_dim,
        inner_dim or args.encoder_embed_dim,
        num_classes,
        args.pooler_activation_fn,
        args.pooler_dropout,
    )
