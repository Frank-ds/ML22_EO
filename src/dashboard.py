import pandas as pd
from streamlit_option_menu import option_menu
import streamlit as st
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
parent_directory = current_directory.parent
directory = parent_directory / 'models' / 'ray'
print(directory)

directories = ds.get_directories(directory)
print(directories)

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

    if st.button("Get Hypertune Data"):
        plot,p = ds.get_hypertune_data(selected_directory, directory)
        plot_df = pd.DataFrame(p)
        st.dataframe(plot_df)
        print(plot_df.columns) 
        fig = px.parallel_coordinates(plot_df, color="Accuracy", color_continuous_scale="Blackbody")
        fig.update_layout(width=1100, height=600)
        st.plotly_chart(fig)

########################################Day tab########################################



######################################## GPS tab########################################
if selected == "vizualisations":
    st.title(f" {selected}")

def get_hypertune_data():
    tune_dir2 = Path("models/ray/train_2023-06-08_09-25-44").resolve()
    tune_dir2.exists()
    analysis = ExperimentAnalysis(tune_dir2)

    plot = analysis.results_df
    select = ["Accuracy", "config/hidden_size", "config/dropout", "config/num_layers"]
    p2 = plot[select].reset_index().dropna()
    p2.sort_values("Accuracy", inplace=True)
    return p2


    # GPS plot
    st.subheader("GPS plot")
    fig2 = px.scatter_mapbox(
        df_filtered,
        lat="lat",
        lon="lon",
        hover_data=["vehicle_speed", "lift_height"],
        color="truck_state",
        color_continuous_scale=[
            (0, "Green"),
            (0.5, " yellow"),
            (0.7, "red"),
            (1, "Purple"),
        ],
        zoom=15,
        height=900,
    )
    fig2.update_layout(
        mapbox_style="open-street-map",
        mapbox_layers=[
            {
                "below": "traces",
                "sourcetype": "raster",
                "sourceattribution": "United States Geological Survey",
                "source": [
                    "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
                ],
            }
        ],
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
    )
    st.plotly_chart(fig2, theme="streamlit", use_container_width=True)