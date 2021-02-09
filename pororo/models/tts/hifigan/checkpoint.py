# Copyright (c) Kakao Enterprise, its affiliates. All Rights Reserved

import glob
import os

import torch


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    checkpoint_dict = torch.load(filepath, map_location=device)
    return checkpoint_dict


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + "*")
    cp_list = glob.glob(pattern)
    if not len(cp_list):
        return ""
    return sorted(cp_list)[-1]
