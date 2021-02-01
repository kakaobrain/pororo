# Copyright (c) Facebook, Inc. and its affiliates and Kakao Brain. All Rights Reserved

# flake8: noqa

import glob
import os

from pororo.models.brainbert.BrainLaBERTa import RobertaLabelModel
from pororo.models.brainbert.BrainRoBERTa import BrainRobertaModel
from pororo.models.brainbert.CharBrainRoBERTa import CharBrainRobertaModel
from pororo.models.brainbert.EnBERTa import CustomRobertaModel
from pororo.models.brainbert.JaBERTa import JabertaModel
from pororo.models.brainbert.PoSLaBERTa import RobertaSegmentModel
from pororo.models.brainbert.PosRoBERTa import PosRobertaModel
from pororo.models.brainbert.utils import CustomChar
from pororo.models.brainbert.ZhBERTa import ZhbertaModel


def __import_user_modules():
    """
    Import important packages before using related sources
    """
    package_dir = os.path.dirname(os.path.abspath(__file__))
    reject_codes = ["__init__.py"]
    module_path_list = [
        path for path in glob.glob(os.path.join(package_dir, "tasks", "*.py"))
        if os.path.basename(path) not in reject_codes
    ]
    module_path_list += [
        path
        for path in glob.glob(os.path.join(package_dir, "criterions", "*.py"))
        if os.path.basename(path) not in reject_codes
    ]

    for module_path in module_path_list:
        module_parent, module_name = os.path.split(module_path)
        module_parent = os.path.basename(module_parent)
        module_name = module_name.split(".")[0]
        exec(
            f"from pororo.models.brainbert.{module_parent} import {module_name}"
        )


__import_user_modules()
