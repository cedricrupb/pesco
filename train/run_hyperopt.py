import os
import sys

lib_dir = os.path.join(os.path.dirname(__file__), "..", "lib", "python")

sys.dont_write_bytecode = True  # prevent creation of .pyc files
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pesco")) # Add PeSCo also
sys.path.insert(0, lib_dir)

if "PYTHONPATH" not in os.environ:
    os.environ["PYTHONPATH"] = ""
os.environ["PYTHONPATH"] += os.pathsep + str(
    lib_dir
)  # necessary so subprocesses also use libraries


import os
import argparse

import optuna

import json
import random
import numpy as np

from copy import copy

from run_optim_selection import args_to_config, TrainConfig

from pesco.data.utils import load_dataset
from pesco.eval       import load_evaluator_from_file
from run_optim_selection import clean_dataset, filter_dataset, crossval


def main():
    config = args_to_config(no_check = True)

    optimized_params = {}
    for k, v in config.__dict__.items():
        if isinstance(v, str) and v.startswith("["):
            v = json.loads(v)
            optimized_params[k] = v

    if len(optimized_params) == 0:
        print("No parameter to optimize")
        return

    # Set the seed
    random.seed(config.seed)
    np.random.seed(config.seed)

    print("Current config:", config)
   
    print("Load dataset...")
    dataset = load_dataset(
        config.feature_path, config.label_path, fill_unknown = True
    )

    if len(config.index) > 0:
        dataset = filter_dataset(config, dataset)

    if config.timelimit != 900:
        runtime_mask = np.clip(config.timelimit - dataset.runtimes, 0, 1)
        labels  = dataset.labels * runtime_mask.astype(np.int64)
        dataset = Dataset(dataset[0], labels, *dataset[2:])

    dataset = clean_dataset(dataset)

    print(f"Loaded {dataset.embedding.shape[0]} instances.")
    print(f"Embeddings have {dataset.embedding.shape[1]} dimensions.")
    print(f"Ranked verifiers: {dataset.label_index}")

    evaluator = load_evaluator_from_file(config.label_path)

    def run_crossval(trial):
        nconfig = copy(config)

        for k, v in optimized_params.items():
            param = trial.suggest_categorical(k, v)
            setattr(nconfig, k, param)

        dummy = TrainConfig()
        for k, v in dummy.__dict__.items():
            vtype = type(v)
            setattr(nconfig, k, vtype(getattr(nconfig, k)))
        
        score, _ = crossval(nconfig, evaluator, dataset)
        return -score

    study = optuna.create_study()
    study.optimize(run_crossval, n_trials = 20)

    print(study.best_params)

    

if __name__ == "__main__":
    main()