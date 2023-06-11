import os
from pathlib import Path
import streamlit as st
from ray.tune import ExperimentAnalysis
import ray

current_directory = Path(__file__).resolve().parent
ray_dir = current_directory.parent / 'models' / 'ray'

st.cache_data
def get_directories(path:str):
    directories = []
    for entry in Path(path).iterdir():
        if entry.is_dir():
            directories.append(entry.name)
    return directories

st.cache_data
def get_hypertune_data(prefix:Path,dir_1:Path):
    tune_dir = prefix.joinpath(dir_1)
    exists = tune_dir.exists()

    if exists:
        ray.init(ignore_reinit_error=True)
        analysis = ExperimentAnalysis(str(tune_dir))
        df = analysis.results_df
        select = ["Accuracy", "config/hidden_size", "config/dropout", "config/num_layers","train_loss","test_loss","time_total_s"]
        p = df[select].reset_index().dropna()
        p.sort_values("Accuracy", inplace=True, ascending=False)
        return df,p
    else:
        print(f"Directory '{tune_dir}' does not exist.")
        return None, None

