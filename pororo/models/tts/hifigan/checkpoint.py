import torch
import os
import glob


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    checkpoint_dict = torch.load(filepath, map_location=device)
    return checkpoint_dict


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]
