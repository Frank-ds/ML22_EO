import os
from pathlib import Path
import streamlit as st
from ray.tune import ExperimentAnalysis
import ray
from typing import List, Dict
import pandas as pd
import numpy as np

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
def get_data(prefix: Path, dir_list: List[Path], columns: List[str]):
    dataframes = {}
    for dir_1 in dir_list:
        tune_dir = prefix / Path(dir_1)
        exists = tune_dir.exists()

        if exists:
            ray.init(ignore_reinit_error=True)
            analysis = ExperimentAnalysis(str(tune_dir))
            df = analysis.results_df[columns]  # Select specific columns
            clean = df.reset_index().dropna()
            clean.sort_values("Accuracy", inplace=True, ascending=False)
            dataframes[dir_1] = clean
        else:
            print(f"Directory '{tune_dir}' does not exist.")

    return dataframes

@st.cache_resource
def accuracy(df):
    max_accuracy = round(np.max(df.Accuracy),2)
    mean_accuracy = round(np.mean(df.Accuracy),2)
    total_time = round(np.sum(df.time_total_s)/60,0)
    train_loss = round(np.min(df.train_loss),2)
    test_loss = round(np.min(df.test_loss),2)

    return max_accuracy,mean_accuracy,total_time,train_loss,test_loss

