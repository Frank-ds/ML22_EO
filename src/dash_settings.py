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

def get_hypertune_data(prefix,dir_1):
    tune_dir = Path(prefix) / Path(dir_1).resolve() 
    exists = tune_dir.exists()
    print(tune_dir)

    if exists:
        ray.init(ignore_reinit_error=True)
        analysis = ExperimentAnalysis(str(tune_dir))
        plot = analysis.results_df
        select = ["Accuracy", "config/hidden_size", "config/dropout", "config/num_layers"]
        p = plot[select].reset_index().dropna()
        p.sort_values("Accuracy", inplace=True, ascending=False)
        return plot,p
    else:
        print(f"Directory '{tune_dir}' does not exist.")
        return None, None



