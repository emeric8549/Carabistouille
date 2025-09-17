import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import torch
from models import RNNModel, GRUModel, LSTMModel, CNN1DModel

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    X_train, y_train = train_df.iloc[:, :-1], train_df.iloc[:, -1]
    X_test, y_test = test_df.iloc[:, :-1], test_df.iloc[:, -1]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    encoder = OneHotEncoder(sparse_output=False)
    y_train = encoder.fit_transform(y_train.values.reshape(-1, 1))
    y_test = encoder.transform(y_test.values.reshape(-1, 1))

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    return X_train_t, y_train_t, X_test_t, y_test_t


def get_model(model_name, input_size, hidden_size, num_classes):
    if model_name == "rnn":
        return RNNModel(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
    elif model_name == "gru":
        return GRUModel(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
    elif model_name == "lstm":
        return LSTMModel(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
    elif model_name == "cnn1d":
        return CNN1DModel(hidden_size=hidden_size, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")