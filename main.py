from pipeline import LSTMPipeline

def main():
    pipeline = LSTMPipeline(".../sensor.csv")
    # https://www.kaggle.com/datasets/nphantawee/pump-sensor-data/data

    pipeline.load_data()
    pipeline.preprocess_data()

    X_train, X_test, X_M_test, y_train, y_test, y_M_test = pipeline.split_data()

    pipeline.load_model()

    test_accuracy, test_report, M_test_accuracy, M_test_report, cm = pipeline.evaluate_model(X_test, y_test, X_M_test, y_M_test)

    print(f"Test Set Accuracy: {test_accuracy:.4f}")
    print(test_report)
    print(f"M_test Set Accuracy: {M_test_accuracy:.4f}")
    print(M_test_report)
    print("Confusion Matrix:")
    print(cm)


if __name__ == "__main__":
    main()
