import pandas as pd
import numpy as numpy
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse

from models import RNNModel, GRUModel, LSTMModel, CNN1DModel

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    X_train, y_train = train_df.iloc[:, :-1], train_df.iloc[:, -1]
    X_test, y_test = test_df.iloc[:, :-1], test_df[:, -1]

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


def get_model(model_name, input_size, num_classes):
    if model_name == "rnn":
        return RNNModel(input_size=input_size, num_classes=num_classes)
    elif model_name == "gru":
        return GRUModel(input_size=input_size, num_classes=num_classes)
    elif model_name == "lstm":
        return LSTMModel(input_size=input_size, num_classes=num_classes)
    elif model_name == "cnn1d":
        return CNN1DModel(input_size=input_size, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train(model, train_loader, test_loader, device, epochs=10, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            output = model(X)
            labels = torch.max(y, 1)[1]
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        test_loss = 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                output = model(X)
                labels = torch.max(y, 1)[1]
                predictions = torch.max(output, 1)[1]

                loss = criterion(output, labels)
                test_loss += loss.item()
                total += labels.size(0)
                correct =+ (predictions == labels).sum().item()

        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Training loss: {train_loss/len(train_loader):.4f}, Test loss: {test_loss/len(test_loader):.4f}, Test acc: {acc:.2f}%")

        return model
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["rnn", "gru", "lstm", "cnn1d"], help="rnn | gru | lstm | cnn1d")
    parser.add_argument("--train_path", type=str, default="data/mitbih_train.csv")
    parser.add_argument("--test_path", type=str, default="data/mitbih_test.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")