# Copyright (c) Facebook, Inc. and its affiliates and Kakao Brain. All Rights Reserved

import logging
import os

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from fairseq.data import (
    BaseWrapperDataset,
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    OffsetTokensDataset,
    ReplaceDataset,
    SortDataset,
    data_utils,
)
from fairseq.tasks import FairseqTask, register_task

logger = logging.getLogger(__name__)


@register_task("dependency_parse")
class DependencyParsingLabelTask(FairseqTask):
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
            "--num-classes0",
            type=int,
            default=-1,
            help="number of classes0",
        )
        parser.add_argument("--no-shuffle", action="store_true", default=False)

    def __init__(
        self,
        args,
        data_dictionary,
        pos_dict,
        label0_dictionary,
        label1_dictionary,
    ):
        super().__init__(args)
        self.dictionary = data_dictionary
        self._pos_dictionary = pos_dict
        self._label0_dictionary = label0_dictionary
        self._label1_dictionary = label1_dictionary
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
        assert args.num_classes0 > 0, "Must set --num-classes0"
        assert args.criterion == "dependency_parse"

        # load data dictionary
        data_dict = cls.load_dictionary(
            args,
            os.path.join(args.data, "input0", "dict.txt"),
            source=True,
        )
        logger.info("[input] dictionary: {} types".format(len(data_dict)))

        # load segment dictionary
        pos_dict = cls.load_dictionary(
            args,
            os.path.join(args.data, "input1", "dict.txt"),
            source=False,
        )
        logger.info("[pos] dictionary: {} types".format(len(pos_dict)))

        # load label dictionary
        label0_dict = cls.load_dictionary(
            args,
            os.path.join(args.data, "label0", "dict.txt"),
            source=False,
        )
        logger.info("[label0] dictionary: {} types".format(len(label0_dict)))

        # load label dictionary
        label1_dict = cls.load_dictionary(
            args,
            os.path.join(args.data, "label1", "dict.txt"),
            source=False,
        )
        logger.info("[label1] dictionary: {} types".format(len(label1_dict)))

        return DependencyParsingLabelTask(
            args,
            data_dict,
            pos_dict,
            label0_dict,
            label1_dict,
        )

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
        pos_tokens = make_dataset("input1", self.pos_dictionary)

        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(len(src_tokens))

        label0_dataset = make_dataset("label0", self.label0_dictionary)
        label1_dataset = make_dataset("label1", self.label1_dictionary)

        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": RightPadDataset(
                    src_tokens,
                    pad_idx=self.source_dictionary.pad(),
                    pad_to_length=self._max_positions,
                ),
                "src_lengths": NumelDataset(src_tokens, reduce=False),
            },
            "segments": {
                "seg_tokens": RightPadDataset(
                    pos_tokens,
                    pad_idx=self.pos_dictionary.pad(),
                    pad_to_length=self._max_positions,
                ),
                "seg_lengths": NumelDataset(pos_tokens, reduce=False),
            },
            "target0": RightPadDataset(  # use 1 as padding, will be used to mask out padding when calculating loss
                ReplaceDataset(  # replace eos and existing padding (used when some tokens should not be predicted) with -1
                    OffsetTokensDataset(  # offset tokens to get the targets to the correct range (0,1,2,...)
                        label0_dataset,
                        offset=-self.label0_dictionary.nspecial,
                    ),
                    replace_map={
                        self.label0_dictionary.eos()
                        - self.label0_dictionary.nspecial: -1,
                        self.label0_dictionary.pad()
                        - self.label0_dictionary.nspecial: -1,
                    },
                    offsets=np.zeros(len(label0_dataset), dtype=np.int),
                ),
                pad_idx=-1,
                pad_to_length=self._max_positions,
            ),
            "target1": RightPadDataset(  # use 1 as padding, will be used to mask out padding when calculating loss
                ReplaceDataset(  # replace eos and existing padding (used when some tokens should not be predicted) with -1
                    OffsetTokensDataset(  # offset tokens to get the targets to the correct range (0,1,2,...)
                        label1_dataset,
                        offset=-self.label1_dictionary.nspecial,
                    ),
                    replace_map={
                        self.label1_dictionary.eos()
                        - self.label1_dictionary.nspecial: -1,
                        self.label1_dictionary.pad()
                        - self.label1_dictionary.nspecial: -1,
                    },
                    offsets=np.zeros(len(label1_dataset), dtype=np.int),
                ),
                pad_idx=-1,
                pad_to_length=self._max_positions,
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

        register_dependency_parse_head(
            model,
            args,
            "dependency_parse_head",
            num_classes0=args.num_classes0,
        )
        assert "dependency_parse_head" in model.classification_heads

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
    def pos_dictionary(self):
        return self._pos_dictionary

    @property
    def label0_dictionary(self):
        return self._label0_dictionary

    @property
    def label1_dictionary(self):
        return self._label1_dictionary


"""
The following code should be included in the model definition,
but we keep them here to allow external fairseq installations.
"""


class DependencyParseHead(nn.Module):
    """Head for sequence tagging tasks."""

    def __init__(
        self,
        d_model,
        max_position,
        num_classes0,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation_fn = F.relu
        self.dropout = nn.Dropout(p=dropout)
        self.out_proj = nn.Linear(d_model, num_classes0)

        self.head_attn_pre = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
        )
        self.head_attn_post = nn.MultiheadAttention(
            d_model,
            1,
            dropout=dropout,
        )

    def forward(self, features, masks, **kwargs):
        x = features
        x = self.dropout(x)

        x2 = x.permute(1, 0, 2)
        x3, _ = self.head_attn_pre(x2, x2, x2, key_padding_mask=masks)
        # x3   = [max_len, bsz, hidden_dim]
        _, attn = self.head_attn_post(x3, x3, x3, key_padding_mask=masks)
        # attn = [bsz, max_len, max_len]

        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        label = self.out_proj(x)

        return attn, label


def register_dependency_parse_head(
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
    model.classification_heads[name] = DependencyParseHead(
        args.encoder_embed_dim,
        max_position=args.max_positions,
        num_classes0=args.num_classes0,
    )


class PadDataset(BaseWrapperDataset):

    def __init__(self, dataset, pad_idx, left_pad, pad_to_length):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad
        self.pad_to_length = pad_to_length

    def collater(self, samples):
        return collate_tokens(
            samples,
            self.pad_idx,
            left_pad=self.left_pad,
            pad_to_length=self.pad_to_length,
        )


class RightPadDataset(PadDataset):

    def __init__(self, dataset, pad_idx, pad_to_length):
        super().__init__(
            dataset,
            pad_idx,
            left_pad=False,
            pad_to_length=pad_to_length,
        )


def collate_tokens(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res
