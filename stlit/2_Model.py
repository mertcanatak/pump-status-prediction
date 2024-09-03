import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

url = "https://raw.githubusercontent.com/mertcanatak/pump-status-prediction/main/preprocessed_data.csv"
df = pd.read_csv(url)

if 'show_model' not in st.session_state:
    st.session_state['show_model'] = False 

col1, col2 = st.columns(2)

# Button1 "Show Model"
if col1.button('Show Confusion Matrix'):
    st.session_state['show_model'] = True

# Button2 "Show Statistics"
if col2.button('Show Statistics'):
    st.session_state['show_model'] = False

if st.session_state['show_model']:
    image_path = "https://raw.githubusercontent.com/mertcanatak/pump-status-prediction/main/stlit/photos/cm.jpeg"
    st.image(image_path, caption='Confusion Matrix', use_column_width=True)
else:
    sensor_columns = {
        'sensor_00': 'Motor Casing Vibration',
        'sensor_01': 'Motor Frequency A',
        'sensor_02': 'Motor Frequency B',
        'sensor_03': 'Motor Frequency C',
        'sensor_04': 'Motor Speed',
        'sensor_05': 'Motor Current',
        'sensor_06': 'Motor Active Power',
        'sensor_07': 'Motor Apparent Power',
        'sensor_08': 'Motor Reactive Power',
        'sensor_09': 'Motor Shaft Power',
        'sensor_10': 'Motor Phase Current A',
        'sensor_11': 'Motor Phase Current B',
        'sensor_12': 'Motor Phase Current C',
        'sensor_40': 'Pump Thrust Bearing Active Temp',
        'sensor_48': 'Pump Inlet Pressure',
        'sensor_49': 'Pump Temp Unknown',
        'sensor_50': 'Pump Discharge Pressure 1',
        'sensor_51': 'Pump Discharge Pressure 2'}

    selected_sensors = st.multiselect("Please select at least one sensor (maximum 3):",
                                      list(sensor_columns.keys()),
                                      format_func=lambda x: sensor_columns[x],
                                      default=["sensor_01"])

    if len(selected_sensors) > 3:
        st.warning("You can select a maximum of 3 sensors.")
    elif len(selected_sensors) > 0:
        df_downsampled = df[::20] 

        # Convert DataFrame to NumPy array for faster operations
        sensor_data = df_downsampled[selected_sensors].values
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(sensor_data)

        df_scaled = pd.DataFrame(scaled_data, columns=selected_sensors)
        df_scaled["machine_status"] = df_downsampled["machine_status"].values

        fig, ax = plt.subplots(figsize=(12, 8)) 

        fig.patch.set_facecolor('#A7C7E7')
        ax.set_facecolor('#F0F0F0')

        for sensor in selected_sensors:
            ax.plot(df_scaled.index, df_scaled[sensor], label=sensor_columns[sensor], linewidth=1)

        ax.plot(df_scaled.index, df_scaled["machine_status"], label="Machine Status", color='#0C005C', linewidth=1.5)
        ax.fill_between(df_scaled.index, df_scaled["machine_status"], color='#0C005C', alpha=0.4)

        
        ax.set_ylim(-5, 5)
        ax.set_xlabel("Index")
        ax.set_ylabel("Values")
        ax.set_title("Sensor Data vs Machine Status", fontsize=18)
        ax.legend()

        st.pyplot(fig)
    else:
        st.warning("Please select at least one sensor!")
