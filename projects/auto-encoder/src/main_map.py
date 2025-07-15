from utils import get_mnist
from train import train
from model import Autoencoder

import torch
import torch.nn as nn
from torch.optim import Adam

if __name__ == "__main__":
    batch_size = 256
    n_epochs = 5
    lr = 1e-3

    dataloader_train, dataloader_test = get_mnist(batch_size, shuffle=True, download=True)

    input_channels = next(iter(dataloader_train))[0].shape[1]
    output_channels = input_channels
    model = Autoencoder(input_channels=input_channels, output_channels=output_channels)

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train(model, dataloader_train, dataloader_test, optimizer, criterion, n_epochs, device)

    torch.save(model.state_dict(), "model.pth")