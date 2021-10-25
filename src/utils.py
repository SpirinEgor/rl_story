import random
from typing import Dict

import numpy
from ray.rllib.agents import ppo


def fix_everything(seed: int):
    random.seed(seed)
    numpy.random.seed(seed)


def build_ppo_config(config: Dict) -> Dict:
    train_config = ppo.DEFAULT_CONFIG.copy()
    for key, value in config["ppo"].items():
        train_config[key] = value
    # fix conv filters
    conv_filters = []
    for conv in train_config["model"]["conv_filters"]:
        kernel_size = int(conv[1][1:]), int(conv[2][:-1])
        conv_filters.append([conv[0], kernel_size, conv[3]])
    train_config["model"]["conv_filters"] = conv_filters
    return train_config
