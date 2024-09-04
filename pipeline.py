import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam


class LSTMPipeline:
    def __init__(self, data_path,
                 model_path="C:/Users/Mert Can/PycharmProjects/pythonProject/lstm_model.keras"):
        self.data_path = data_path
        self.model_path = model_path
        self.sensor_columns = ['sensor_00', 'sensor_01', 'sensor_02', 'sensor_03', 'sensor_04',
                               'sensor_05', 'sensor_06', 'sensor_07', 'sensor_08', 'sensor_09',
                               'sensor_10', 'sensor_11', 'sensor_12', 'sensor_40', 'sensor_48',
                               'sensor_49', 'sensor_50', 'sensor_51']
        self.df = None
        self.X = None
        self.y = None
        self.model = None

    def load_data(self):
        self.df = pd.read_csv(self.data_path)

    def preprocess_data(self):
        # machine_status mapping
        status_mapping = {"NORMAL": 0, "RECOVERING": 1, "BROKEN": 1}
        self.df['machine_status'] = self.df['machine_status'].map(status_mapping)

        self.df[self.sensor_columns] = self.df[self.sensor_columns].apply(lambda x: x.fillna(x.median()))

        # 24 hours
        self.df['shifted_machine_status'] = self.df['machine_status'].shift(-1440)
        self.df = pd.concat([self.df[self.sensor_columns], self.df["shifted_machine_status"]], axis=1)

        self.df.dropna(inplace=True, axis=0)
        self.df.reset_index(drop=True, inplace=True)

        # X ve y
        self.X = self.df[self.sensor_columns].values.reshape((self.df.shape[0], len(self.sensor_columns), 1))
        self.y = self.df['shifted_machine_status'].values

    def split_data(self):
        X_train, X_temp, y_train, y_temp = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        X_test, X_M_test, y_test, y_M_test = train_test_split(X_temp, y_temp, test_size=2 / 3, random_state=42)
        return X_train, X_test, X_M_test, y_train, y_test, y_M_test

    def load_model(self):
        self.model = load_model(self.model_path)

    def evaluate_model(self, X_test, y_test, X_M_test, y_M_test):
        y_pred = (self.model.predict(X_test) > 0.5).astype(int)
        test_accuracy = accuracy_score(y_test, y_pred)
        test_report = classification_report(y_test, y_pred)

        y_pred_M_test = (self.model.predict(X_M_test) > 0.5).astype(int)
        M_test_accuracy = accuracy_score(y_M_test, y_pred_M_test)
        M_test_report = classification_report(y_M_test, y_pred_M_test)

        cm = confusion_matrix(y_M_test, y_pred_M_test)

        return test_accuracy, test_report, M_test_accuracy, M_test_report, cm
