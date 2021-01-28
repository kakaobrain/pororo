# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from torch import BoolTensor, Tensor


def get_mask_from_lengths(inputs: Tensor, seq_lengths: Tensor) -> Tensor:
    mask = BoolTensor(inputs.size()).fill_(False)

    for idx, length in enumerate(seq_lengths):
        length = length.item()

        if (mask[idx].size(0) - length) > 0:
            mask[idx].narrow(
                dim=0,
                start=length,
                length=mask[idx].size(0) - length,
            ).fill_(True)

    return mask


def collate_fn(batch: list, batch_size: int) -> dict:

    def seq_length_(p):
        return len(p)

    input_lengths = torch.IntTensor([len(s) for s in batch])
    max_seq_length = max(batch, key=seq_length_).size(0)
    inputs = torch.zeros(batch_size, max_seq_length)

    for idx in range(batch_size):
        sample = batch[idx]
        input_length = sample.size(0)
        inputs[idx].narrow(dim=0, start=0, length=input_length).copy_(sample)

    return {"inputs": inputs, "input_lengths": input_lengths}
