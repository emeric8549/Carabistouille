import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=5, dropout=0.2):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=5, dropout=0.2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=5, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class CNN1DModel(nn.Module):
    def __init__(self, input_channels=1, hidden_size=16, num_classes=5):
        super(CNN1DModel, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, hidden_size, kernel_size=20, padding=2)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden_size * 2)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x