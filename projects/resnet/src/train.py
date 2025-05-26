from model import ResNet34
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.optim import SGD

from torchvision.datasets import CIFAR10
from torchvision import transforms


transform = transforms.Compose([transforms.Resize(256),
                                transforms.RandomHorizontalFlip(0.5),
                                transforms.RandomCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # We use the default per_pixel normalization based on the ImageNet dataset
                                ])


data_train = CIFAR10(root='./dataset/train',
                     train=True,
                     transform=transform,
                     download=True)

data_test = CIFAR10(root='./dataset/test',
                     train=False,
                     transform=transform,
                     download=True)

dataloader_train = DataLoader(data_train,
                              batch_size=256,
                              shuffle=True,
                              num_workers=8)

dataloader_test = DataLoader(data_test,
                              batch_size=256,
                              shuffle=True,
                              num_workers=8)




if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet = ResNet34(input_channels=3, output_channels=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(resnet.parameters(), weight_decay=1e-4, momentum=0.9)
    
    for epoch in range(100):
        losses = []
        for x, y in dataloader_train:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            pred = resnet.forward(x)
            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        
        if (epoch+1) % 10 == 0:
            losses_test = []
            for x, y in dataloader_test:
                x, y = x.to(device), y.to(device)
                pred = resnet(x)
                losses_test.append(criterion(pred, y).item())

            print(f"Epoch {epoch+1} | Train loss: {np.mean(np.array(losses)):4f} | Test loss: {np.mean(np.array(losses_test))}")