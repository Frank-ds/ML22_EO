import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pathlib import Path
from ray.tune import ExperimentAnalysis
import ray
import dash_settings as ds
import sys

# sys.path.insert(0, "..")

current_directory = Path(__file__).resolve().parent
ray_dir = current_directory.parent / 'models' / 'ray'
directories = ds.get_directories(ray_dir)

columns = ["Accuracy", "config/hidden_size", "config/dropout", "config/num_layers", "train_loss", "test_loss", "time_total_s"]
dataframes = ds.get_data(ray_dir, directories, columns)

######################################## Dashboard ########################################
st.set_page_config(layout="wide", initial_sidebar_state="expanded")


def streamlit_menu():
    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options=["Hypertune results", "Metrics"],
            icons=["sliders", "bar-chart-line", "geo"],
            menu_icon="clipboard-data",
            default_index=0,
        )
    return selected


######################################## Home tab ########################################
selected = streamlit_menu()

if selected == "Hypertune results":
    st.title(f" {selected}")

    selected_directory = st.selectbox("Select a experiment", directories)
    st.write(selected_directory)

    if st.button("Get Hypertune Data"):
        selected_df = dataframes[selected_directory]

        max,mean,total_time,trainloss,testloss = ds.accuracy(selected_df)
        
        st.markdown("### Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("max_accuracy",f'{max} %')
        col2.metric("mean_accuracy",f'{mean} %')
        col3.metric("total time",f'{total_time}min')
        col4.metric("train loss",trainloss)
        col5.metric("test loss",testloss)

        
        fig = px.parallel_coordinates(selected_df, color="Accuracy", color_continuous_scale=[(0.00, "red"),   (0.50, "red"),
                                                     (0.50, "orange"), (0.90, "orange"),
                                                     (0.95, "green"),  (1.00, "green")])
        fig.update_layout(width=1100, height=600)
        st.plotly_chart(fig)


######################################## GPS tab########################################
if selected == "Metrics":
    st.title(f" {selected}")
    
    selected_directory = st.selectbox("Select a directory", directories)

    if st.button("Get DataFrame"):
        selected_df = dataframes[selected_directory]
        st.dataframe(selected_df)


        

