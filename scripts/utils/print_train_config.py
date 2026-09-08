#!/usr/bin/env python3
"""Resolve the real Hydra training recipe without importing CUDA, Ray or verl."""
import os
from pathlib import Path
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]


def main():
    os.chdir(ROOT)
    with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base=None):
        config = compose(config_name='train', overrides=sys.argv[1:])
    OmegaConf.resolve(config)
    workers = config.trainer.n_gpus_per_node
    tp = config.actor_rollout_ref.rollout.tensor_model_parallel_size
    if not isinstance(workers, int) or not isinstance(tp, int) or workers < 1 or tp < 1 or workers % tp:
        raise ValueError('GPU worker count and rollout TP must be positive integers; TP must divide the worker count')
    print(OmegaConf.to_yaml(config, resolve=True))


if __name__ == '__main__':
    main()
