"""There are three configurable things: predictors/models, data, training/evaluation.

Only first two require factories.
"""
import copy
import os

import pandas as pd
import torch
from src import constants, factories
from src.domain_chop import PairwiseDomainPredictor
from src.loggers import get_versioned_dir
from src.models.rosetta import trRosettaNetwork
from src.domain_assignment.assigners import SparseLowRank
from src.utils.torch_utils import get_torch_device
from src.utils import common as common_utils


def get_assigner(config):
    assigner_type = config["type"]
    if assigner_type == "sparse_lowrank":
        assigner = SparseLowRank(**config["kwargs"])
    else:
        return ValueError()
    return assigner


def get_model(config):
    model_type = config["type"]
    if model_type == "trrosetta":
        model = trRosettaNetwork(**config["kwargs"])
    else:
        return ValueError()
    return model


def pairwise_predictor(learner_config, force_cpu=False, output_dir=None):
    model = get_model(learner_config["model"])
    assigner = get_assigner(learner_config["assignment"])
    device = get_torch_device(force_cpu=force_cpu)
    model.to(device)
    kwargs = {k: v for k, v in learner_config.items() if k not in ["model", "assignment"]}
    print("Learner kwargs", kwargs, flush=True)
    return PairwiseDomainPredictor(model, assigner, device, checkpoint_dir=output_dir, **kwargs)


def pretrained_predictor(
    experiment_name,
    output_dir,
    version=None,
    use_versioning=True,
    average=False,
    old_style=False,  # before weight averaging support model checkpoints were handled differently
    data_dir_map_from=None,
    use_best_val=False,
):
    """Load a saved checkpoint associated with experiment_name.

    N.B. doesn't average checkpoints by default, even if available.
    """
    output_dir = os.path.join(output_dir, experiment_name)
    if use_versioning:
        # version = None will load last version
        output_dir, _ = get_versioned_dir(output_dir, version=version, resume=True)
    config = common_utils.load_json(os.path.join(output_dir, "config.json"))
    if use_best_val:
        output_dir = os.path.join(output_dir, "best_val")
    learner = factories.pairwise_predictor(config["learner"], output_dir=output_dir)
    learner.eval()
    learner.load_checkpoints(average=average, old_style=old_style)
    if data_dir_map_from is not None:
        # override saved DATA_DIR to local value in relevant config fields
        # e.g. data_dir_map_from = /SAN/bioinf/domdet
        config["data"] = {
            k: v.replace(data_dir_map_from, constants.DATA_DIR) if isinstance(v, str) else v
            for k, v in config["data"].items()
        }

    return learner, config


def get_test_ids(label_path, feature_path, csv_path=None):
    ids = [id.split('.')[0] for id in set(os.listdir(label_path)).intersection(set(os.listdir(feature_path)))]
    if csv_path is not None:
        df = pd.read_csv(csv_path)
        ids = sorted(list(set(ids).intersection(set(df.chain_id))))
    return ids


def filter_plddt(df_path, ids, threshold=90):
    df = pd.read_csv(df_path)
    df = df[df.plddt > threshold]
    ids = [i for i in ids if i in df.casp_id.values]
    return ids


