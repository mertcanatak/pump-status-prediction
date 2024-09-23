# Project Explanation
This project uses an LSTM model to predict pump status based on sensor data, aiming to assess the performance of both general and specific subsets of test data. 

## Project Structure

**main.py:** Executes the model for prediction.

**pipeline.py:** Data preprocessing and model pipeline.

**requirements.txt:** Contains necessary dependencies.

**lstm_model.keras:** Pretrained LSTM model.

You can check the app here: [Pump Status Prediction App](https://pump-status-prediction-apybusvusrkgebgbjwhqwh.streamlit.app/)

You can view detailed visualizations, such as graphs and confusion matrices, which represent the model's performance. The site provides to evaluate the model's predictions in real-time and offers an intuitive way to analyze the pump system status based on uploaded data.

## Data Loading & Preprocessing:
The dataset is loaded from a CSV file. (Kaggle dataset: [Pump Sensor Data](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data/data) ).

The data is then preprocessed, ensuring it's in the correct format for training.

## Train-Test Split:
The data is split into training, testing, and a subset (M_test) for specialized evaluation.

Model Loading & Evaluation:

A pre-trained LSTM model is loaded to make predictions on the test sets.
The model's performance is evaluated using accuracy, classification reports, and confusion matrices for both the standard and subset test data.

## Results:
Test set accuracy and classification metrics are printed for interpretation.
This approach ensures that the model's robustness is tested on different subsets, making the predictions more reliable.
![CM](https://github.com/mertcanatak/pump-status-prediction/blob/main/stlit/photos/cm.jpeg)
