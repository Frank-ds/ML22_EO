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


######################################## Dashboard ########################################
st.set_page_config(layout="wide", initial_sidebar_state="expanded")


def streamlit_menu():
    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options=["Hypertune parameters", "vizualisations"],
            icons=["sliders", "bar-chart-line", "geo"],
            menu_icon="clipboard-data",
            default_index=0,
        )
    return selected


######################################## Home tab ########################################
selected = streamlit_menu()

if selected == "Hypertune parameters":
    st.title(f" {selected}")

    selected_directory = st.selectbox("Select a directory", directories)
    st.write(selected_directory)

    if st.button("Get Hypertune Data"):
        df, p = ds.get_hypertune_data(ray_dir, selected_directory)
        plot_df = pd.DataFrame(p)
        st.dataframe(plot_df)
        fig = px.parallel_coordinates(plot_df, color="Accuracy", color_continuous_scale=[(0.00, "red"),   (0.50, "red"),
                                                     (0.50, "orange"), (0.90, "orange"),
                                                     (0.95, "green"),  (1.00, "green")])
        fig.update_layout(width=1100, height=600)
        st.plotly_chart(fig)


########################################Day tab########################################



######################################## GPS tab########################################
if selected == "vizualisations":
    st.title(f" {selected}")
    


    st.markdown("### Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy",1)
    col2.metric("Hidden size",2)
    col3.metric("Number layers",3)
    col4.metric("dropout",4)
    col5.metric("loss",5
    )

