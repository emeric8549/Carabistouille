from model import UNet
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD

from torchvision.datasets import VOCSegmentation
from torchvision import transforms

transform = transforms.Compose([transforms.Resize(256),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # We use the default per_pixel normalization based on the ImageNet dataset
                                ])

target_transform = transforms.Compose([transforms.Resize(572),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                       ])

data_train = VOCSegmentation(root='./dataset/train',
                     image_set='train',
                     transform=transform,
                     target_transform=target_transform,
                     download=True)

data_test = VOCSegmentation(root='./dataset/test',
                     image_set='test',
                     year='2007',
                     transform=transform,
                     target_transform=target_transform
                     download=True)

dataloader_train = DataLoader(data_train,
                              batch_size=16,
                              shuffle=True,
                              num_workers=8)

dataloader_test = DataLoader(data_test,
                              batch_size=16,
                              shuffle=True,
                              num_workers=8)




if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    unet = UNet(input_channels=3).to(device)
    print(f"Number of parameters: {sum(p.numel() for p in unet.parameters() if p.requires_grad)}")
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(unet.parameters(), weight_decay=1e-4, momentum=0.9, lr=0.1)
    
    for epoch in range(100):
        losses = []
        for x, y in tqdm(dataloader_train, desc="Training"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            pred = unet(x)
            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if (epoch+1) % 1 == 0:
            losses_test = []
            for x, y in tqdm(dataloader_test, desc="Testing"):
                x, y = x.to(device), y.to(device)
                pred = unet(x)
                losses_test.append(criterion(pred, y).item())

            print(f"Epoch {epoch+1} | Train loss: {np.mean(np.array(losses)):4f} | Test loss: {np.mean(np.array(losses_test))}")