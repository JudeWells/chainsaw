import copy
import json
import subprocess

import logging
LOG = logging.getLogger(__name__)

def load_json(jsonfile):
    with open(jsonfile, "r") as jf:
        res = json.load(jf)
    return res


def save_config(config, filepath):
    with open(filepath, "w") as outfile:    
        json.dump(config, outfile, indent=4)


def apply_diff(base_cfg, diff):
    cfg = copy.deepcopy(base_cfg)
    for k, v in diff.items():
        if isinstance(v, dict):
            cfg[k] = apply_diff(base_cfg[k], v)
        else:
            cfg[k] = v
    return cfg

def execute_bash_command(bash_command_string):
    bash_return = subprocess.run(bash_command_string.split(), timeout=20)
    return bash_return