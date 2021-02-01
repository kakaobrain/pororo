# Copyright (c) Facebook, Inc. and its affiliates and Kakao Brain. All Rights Reserved

import logging
import os

import numpy as np
import torch.nn as nn
from fairseq import utils
from fairseq.data import (
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    OffsetTokensDataset,
    ReplaceDataset,
    RightPadDataset,
    SortDataset,
    data_utils,
)
from fairseq.tasks import FairseqTask, register_task

logger = logging.getLogger(__name__)


@register_task("sequence_tagging")
class SequenceTaggingTask(FairseqTask):
    """
    Sequence tagging (also called sentence tagging or sequence labelling) task that predicts a class for each input token.
    Inputs should be stored in 'input' directory, labels in 'label' directory.
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
            default=-1,
            help="number of classes",
        )
        parser.add_argument("--no-shuffle", action="store_true", default=False)

    def __init__(self, args, data_dictionary, label_dictionary):
        super().__init__(args)
        self.dictionary = data_dictionary
        self._label_dictionary = label_dictionary
        if not hasattr(args, "max_positions"):
            self._max_positions = (
                args.max_source_positions,
                args.max_target_positions,
            )
        else:
            self._max_positions = args.max_positions
        args.tokens_per_sample = self._max_positions

    @classmethod
    def load_dictionary(cls, args, filename, source=True):
        """Load the dictionary from the filename
        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        dictionary.add_symbol("<mask>")
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert args.num_classes > 0, "Must set --num-classes"

        # load data dictionary
        data_dict = cls.load_dictionary(
            args,
            os.path.join(args.data, "input0", "dict.txt"),
            source=True,
        )
        logger.info("[input] dictionary: {} types".format(len(data_dict)))

        # load label dictionary
        label_dict = cls.load_dictionary(
            args,
            os.path.join(args.data, "label", "dict.txt"),
            source=False,
        )
        logger.info("[label] dictionary: {} types".format(len(label_dict)))
        return SequenceTaggingTask(args, data_dict, label_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        def get_path(type, split):
            return os.path.join(self.args.data, type, split)

        def make_dataset(type, dictionary):
            split_path = get_path(type, split)

            dataset = data_utils.load_indexed_dataset(
                split_path,
                dictionary,
                self.args.dataset_impl,
                combine=combine,
            )
            assert dataset is not None, "could not find dataset: {}".format(
                get_path(type, split))
            return dataset

        src_tokens = make_dataset("input0", self.source_dictionary)

        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(len(src_tokens))

        label_dataset = make_dataset("label", self.label_dictionary)

        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": RightPadDataset(
                    src_tokens,
                    pad_idx=self.source_dictionary.pad(),
                ),
                "src_lengths": NumelDataset(src_tokens, reduce=False),
            },
            "target": RightPadDataset(  # use 1 as padding, will be used to mask out padding when calculating loss
                ReplaceDataset(  # replace eos and existing padding (used when some tokens should not be predicted) with -1
                    OffsetTokensDataset(  # offset tokens to get the targets to the correct range (0,1,2,...)
                        label_dataset,
                        offset=-self.label_dictionary.nspecial,
                    ),
                    replace_map={
                        self.label_dictionary.eos()
                        - self.label_dictionary.nspecial: -1,
                        self.label_dictionary.pad()
                        - self.label_dictionary.nspecial: -1,
                    },
                    offsets=np.zeros(len(label_dataset), dtype=np.int),
                ),
                pad_idx=-1,
            ),
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens, reduce=True),
        }

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

        register_sequence_tagging_head(
            model,
            args,
            "sequence_tagging_head",
            num_classes=args.num_classes,
        )
        assert "sequence_tagging_head" in model.classification_heads

        return model

    def max_positions(self):
        return self._max_positions

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    @property
    def label_dictionary(self):
        return self._label_dictionary


"""
The following code should be included in the model definition,
but we keep them here to allow external fairseq installations.
"""


class SequenceTaggingHead(nn.Module):
    """Head for sequence tagging tasks."""

    def __init__(self, input_dim, inner_dim, num_classes, activation_fn,
                 pooler_dropout):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


def register_sequence_tagging_head(
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
    model.classification_heads[name] = SequenceTaggingHead(
        args.encoder_embed_dim,
        inner_dim or args.encoder_embed_dim,
        num_classes,
        args.pooler_activation_fn,
        args.dropout,
    )
