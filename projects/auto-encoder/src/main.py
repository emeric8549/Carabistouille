from utils import get_data_blurred, get_data_colorized, imcompare
from train import train
from model import Autoencoder

import torch
import torch.nn as nn
from torch.optim import Adam

if __name__ == "__main__":
    batch_size = 1024
    n_epochs = 20
    lr = 1e-3
    
    dataloader_train, dataloader_test = get_data_blurred(batch_size, shuffle=True, download=True)
    input_channels = next(iter(dataloader_train))[0].shape[1]
    output_channels = next(iter(dataloader_train))[1].shape[1]
    model = Autoencoder(input_channels=input_channels, output_channels=output_channels)

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train(model, dataloader_train, dataloader_test, optimizer, criterion, n_epochs, device)


    images_test, images_test_real = next(iter(dataloader_test))
    output = model(images_test.to(device)).cpu().detach()

    imcompare(images_test, output, images_test_real)