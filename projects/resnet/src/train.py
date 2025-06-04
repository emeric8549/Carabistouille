from model import ResNet34
from utils import get_data

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD


if __name__ == '__main__':
    dataloader_train, dataloader_test = get_data(batch_size=256, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    resnet = ResNet34(input_channels=3, output_channels=10).to(device)
    print(f"Number of parameters: {sum(p.numel() for p in resnet.parameters() if p.requires_grad)}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(resnet.parameters(), weight_decay=1e-4, momentum=0.9, lr=0.1)
    
    for epoch in range(100):
        losses = []
        for x, y in tqdm(dataloader_train, desc="Training"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            pred = resnet(x)
            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if (epoch+1) % 1 == 0:
            losses_test = []
            for x, y in tqdm(dataloader_test, desc="Testing"):
                x, y = x.to(device), y.to(device)
                pred = resnet(x)
                losses_test.append(criterion(pred, y).item())

            print(f"Epoch {epoch+1} | Train loss: {np.mean(np.array(losses)):4f} | Test loss: {np.mean(np.array(losses_test))}")