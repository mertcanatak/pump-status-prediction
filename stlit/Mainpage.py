import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import requests

url = "https://raw.githubusercontent.com/mertcanatak/pump-status-prediction/main/preprocessed_data.csv"
df = pd.read_csv(url)

st.set_page_config(layout="wide",
                   initial_sidebar_state="expanded",
                   page_title="Machine Status Prediction")

# Sidebar configuration
st.sidebar.header("Project Overview")
st.sidebar.success("Explore")


page = st.sidebar.radio("Select a page:", ("Overview", "Statistics", "Model"))

if page == "Overview":
    st.title("Machine Status Prediction Using Data-Driven Insights")
    st.header("Overview of the Machine Status Prediction Project")
    st.write("The aim of this project is to predict the operational status of a machine using historical sensor data, leveraging the power of deep learning models to anticipate potential failures and optimize maintenance schedules.")

    if 'show_heatmap' not in st.session_state:
        st.session_state['show_heatmap'] = False 


    col1, col2 = st.columns(2)

    if col1.button('Overview'):
        st.session_state['show_heatmap'] = False

    if col2.button('Show Correlation Heatmap'):
        st.session_state['show_heatmap'] = True


    if st.session_state['show_heatmap']:
        image_path = "https://raw.githubusercontent.com/mertcanatak/pump-status-prediction/main/stlit/photos/corr.jpeg"
        image_caption = "Correlation between Sensors and Machine Status"
    else:
        image_path = "https://raw.githubusercontent.com/mertcanatak/pump-status-prediction/main/stlit/photos/pump-img.jpeg"

    st.image(image_path, caption=image_caption, use_column_width=True)

elif page == "Statistics":
    st.title("Statistics")
    stats_url = "https://raw.githubusercontent.com/mertcanatak/pump-status-prediction/main/stlit/1_Statistics.py"
    response = requests.get(stats_url)
    if response.status_code == 200:
        exec(response.text)  
    else:
        st.error("Failed to load the Statistics page. Please check the URL.")

elif page == "Model":
    st.title("LSTM Model Prediction")
    model_url = "https://raw.githubusercontent.com/mertcanatak/pump-status-prediction/main/stlit/2_Model.py"
    response = requests.get(model_url)
    if response.status_code == 200:
        exec(response.text)
    else:
        st.error("Failed to load the Model page. Please check the URL.")

