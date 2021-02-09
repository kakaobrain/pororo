from collections import OrderedDict

import torch


def lengths_to_mask(lengths, max_length=None):
    """Convert tensor of lengths into a boolean mask."""
    ml = torch.max(lengths) if max_length is None else max_length
    return torch.arange(ml, device=lengths.device)[None, :] < lengths[:, None]


def remove_dataparallel_prefix(state_dict):
    """Removes dataparallel prefix of layer names in a checkpoint state dictionary."""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k[:7] == "module." else k
        new_state_dict[name] = v
    return new_state_dict
