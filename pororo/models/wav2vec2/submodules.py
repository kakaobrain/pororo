# Copyright (c) Facebook, Inc., its affiliates and Kakao Brain. All Rights Reserved

import contextlib
import itertools as it

import torch
import torch.nn as nn
from fairseq.models import FairseqEncoder
from fairseq.models.wav2vec.wav2vec2_asr import (
    Linear,
    Wav2VecCtc,
    base_architecture,
)
from fairseq.tasks.audio_pretraining import AudioPretrainingTask
from wav2letter.criterion import CpuViterbiPath, get_data_ptr_as_bytes
from wav2letter.decoder import CriterionType


class BrainWav2VecEncoder(FairseqEncoder):
    """ Modified from https://github.com/pytorch/fairseq """

    def __init__(self, args, tgt_dict=None, pretrain_args=None):
        self.apply_mask = args.apply_mask

        arg_overrides = {
            "dropout": args.dropout,
            "activation_dropout": args.activation_dropout,
            "dropout_input": args.dropout_input,
            "attention_dropout": args.attention_dropout,
            "mask_length": args.mask_length,
            "mask_prob": args.mask_prob,
            "mask_selection": args.mask_selection,
            "mask_other": args.mask_other,
            "no_mask_overlap": args.no_mask_overlap,
            "mask_channel_length": args.mask_channel_length,
            "mask_channel_prob": args.mask_channel_prob,
            "mask_channel_selection": args.mask_channel_selection,
            "mask_channel_other": args.mask_channel_other,
            "no_mask_channel_overlap": args.no_mask_channel_overlap,
            "encoder_layerdrop": args.layerdrop,
            "feature_grad_mult": args.feature_grad_mult,
        }

        w2v_args = pretrain_args
        assert (args.normalize == w2v_args.normalize
               ), "Fine-tuning works best when data normalization is the same"

        for arg_name, arg_val in arg_overrides.items():
            setattr(args, arg_name, arg_val)

        w2v_args.data = args.data
        task = AudioPretrainingTask.setup_task(w2v_args)
        model = task.build_model(w2v_args)

        model.remove_pretraining_modules()
        super().__init__(task.source_dictionary)

        d = w2v_args.encoder_embed_dim

        self.w2v_model = model

        self.final_dropout = nn.Dropout(args.final_dropout)
        self.freeze_finetune_updates = args.freeze_finetune_updates
        self.num_updates = 0

        if tgt_dict is not None:
            self.proj = Linear(d, len(tgt_dict))
        elif getattr(args, "decoder_embed_dim", d) != d:
            self.proj = Linear(d, args.decoder_embed_dim)
        else:
            self.proj = None

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, tbc=True, **kwargs):
        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            x, padding_mask = self.w2v_model.extract_features(**w2v_args)

            if tbc:
                # B x T x C -> T x B x C
                x = x.transpose(0, 1)

        x = self.final_dropout(x)

        if self.proj:
            x = self.proj(x)

        return {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out[
                "encoder_out"].index_select(1, new_order)
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


class BrainWav2VecCtc(Wav2VecCtc):
    """ Modified from https://github.com/pytorch/fairseq """

    @classmethod
    def build_model(cls, args, target_dict, pretrain_args):
        """Build a new model instance."""
        base_architecture(args)
        w2v_encoder = BrainWav2VecEncoder(args, target_dict, pretrain_args)
        return cls(w2v_encoder, args)


class W2lDecoder(object):

    def __init__(self, tgt_dict):
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)
        self.nbest = 1

        self.criterion_type = CriterionType.CTC
        self.blank = (tgt_dict.index("<ctc_blank>")
                      if "<ctc_blank>" in tgt_dict.indices else tgt_dict.bos())
        self.asg_transitions = None

    def generate(self, models, sample, **unused):
        """Generate a batch of inferences."""
        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v
            for k, v in sample["net_input"].items()
            if k != "prev_output_tokens"
        }
        emissions = self.get_emissions(models, encoder_input)
        return self.decode(emissions)

    def get_emissions(self, models, encoder_input):
        """Run encoder and normalize emissions"""
        encoder_out = models[0](**encoder_input)
        if self.criterion_type == CriterionType.CTC:
            emissions = models[0].get_normalized_probs(
                encoder_out,
                log_probs=True,
            )

        return emissions.transpose(0, 1).float().cpu().contiguous()

    def get_tokens(self, idxs):
        """Normalize tokens by handling CTC blank, ASG replabels, etc."""
        idxs = (g[0] for g in it.groupby(idxs))
        idxs = filter(lambda x: x != self.blank, idxs)

        return torch.LongTensor(list(idxs))


class W2lViterbiDecoder(W2lDecoder):

    def __init__(self, tgt_dict):
        super().__init__(tgt_dict)

    def decode(self, emissions):
        batch_size, time_length, num_classes = emissions.size()

        if self.asg_transitions is None:
            transitions = torch.FloatTensor(
                num_classes,
                num_classes,
            ).zero_()
        else:
            transitions = torch.FloatTensor(self.asg_transitions).view(
                num_classes,
                num_classes,
            )

        viterbi_path = torch.IntTensor(batch_size, time_length)
        workspace = torch.ByteTensor(
            CpuViterbiPath.get_workspace_size(
                batch_size,
                time_length,
                num_classes,
            ))
        CpuViterbiPath.compute(
            batch_size,
            time_length,
            num_classes,
            get_data_ptr_as_bytes(emissions),
            get_data_ptr_as_bytes(transitions),
            get_data_ptr_as_bytes(viterbi_path),
            get_data_ptr_as_bytes(workspace),
        )
        return [[{
            "tokens": self.get_tokens(viterbi_path[b].tolist()),
            "score": 0
        }] for b in range(batch_size)]
