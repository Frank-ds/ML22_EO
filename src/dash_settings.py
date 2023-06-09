import os
from pathlib import Path
from ray.tune import ExperimentAnalysis
import ray


def get_directories(path):
    directories = []
    for entry in Path(path).iterdir():
        if entry.is_dir():
            directories.append(entry.name)
    return directories


def get_hypertune_data(dir, prefix):
    tune_dir = Path(prefix) / Path(dir).resolve()  
    exists = tune_dir.exists()

    if exists:
        analysis = ExperimentAnalysis(tune_dir)
        plot = analysis.results_df
        select = ["Accuracy", "config/hidden_size", "config/dropout", "config/num_layers"]
        return select, plot
    else:
        print(f"Directory '{tune_dir}' does not exist.")
        return None, None



