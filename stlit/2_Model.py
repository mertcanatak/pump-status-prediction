import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import time
import requests

model_url = "https://raw.githubusercontent.com/mertcanatak/pump-status-prediction/main/lstm_model.keras"
X_M_test_url = "https://raw.githubusercontent.com/mertcanatak/pump-status-prediction/main/X_M_test.npy"
y_M_test_url = "https://raw.githubusercontent.com/mertcanatak/pump-status-prediction/main/y_M_test.npy"

def download_file(url, local_path):
    response = requests.get(url)
    with open(local_path, 'wb') as file:
        file.write(response.content)

download_file(model_url, "lstm_model.keras")
download_file(X_M_test_url, "X_M_test.npy")
download_file(y_M_test_url, "y_M_test.npy")

model = load_model("lstm_model.keras")
X_M_test = np.load("X_M_test.npy")
y_M_test = np.load("y_M_test.npy")

sensor_names = [
    'Motor Casing Vibration', 'Motor Frequency A', 'Motor Frequency B', 'Motor Frequency C',
    'Motor Speed', 'Motor Current', 'Motor Active Power', 'Motor Apparent Power',
    'Motor Reactive Power', 'Motor Shaft Power', 'Motor Phase Current A', 'Motor Phase Current B',
    'Motor Phase Current C', 'Pump Thrust Bearing Active Temp', 'Pump Inlet Pressure',
    'Pump Temp Unknown', 'Pump Discharge Pressure 1', 'Pump Discharge Pressure 2']

if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0

st.header("Real-time Data Streaming and Prediction")

start_index = st.number_input("Select Starting Index", min_value=0, max_value=len(X_M_test)-1, value=0, step=1)


if st.button("Set Index"):
    st.session_state.current_index = start_index


col1, col2 = st.columns(2)
if col1.button("Start"):
    st.session_state.is_running = True
if col2.button("Stop"):
    st.session_state.is_running = False

placeholder = st.empty()  

while st.session_state.is_running:
    if st.session_state.current_index < len(X_M_test):
        X_input = X_M_test[st.session_state.current_index].reshape((1, len(X_M_test[0]), 1))
        prediction = model.predict(X_input)

        with placeholder.container():
            col3, col4 = st.columns([3, 1])
            with col3:
                current_values = X_M_test[st.session_state.current_index].flatten()
                st.subheader("Sensor Readings:")
                for i, sensor_name in enumerate(sensor_names):
                    st.write(f"{sensor_name}: {current_values[i]:.2f}")

            with col4:
                st.write(f"Row: {st.session_state.current_index}")
                st.write(f"Predicted Status: {'Normal' if prediction < 0.5 else 'Abnormal'}")
                st.write(f"Actual Status: {'Normal' if y_M_test[st.session_state.current_index] == 0 else 'Abnormal'}")

        st.session_state.current_index += 1

        time.sleep(1)
    else:
        st.write("All data has been processed.")
        st.session_state.is_running = False

if not st.session_state.is_running:
    st.write("Prediction process stopped.")
