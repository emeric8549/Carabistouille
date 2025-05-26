import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD

from torchvision.datasets import CIFAR10
from torchvision import transforms

class ResNetBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or input_channels!=output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(output_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(x))
        out += self.relu(out)


class ResNet34(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ResNet34, self).__init__()

        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=7, padding=3, stride=2)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block1 = nn.Sequential(ResNetBlock(64, 64, stride=1),
                                    ResNetBlock(64, 64, stride=1),
                                    ResNetBlock(64, 64, stride=1))

        self.block2 = nn.Sequential(ResNetBlock(64, 128, stride=2),
                                    ResNetBlock(128, 128, stride=1),
                                    ResNetBlock(128, 128, stride=1),
                                    ResNetBlock(128, 128, stride=1))

        self.block3 = nn.Sequential(ResNetBlock(128, 256, stride=2),
                                    ResNetBlock(256, 256, stride=1),
                                    ResNetBlock(256, 256, stride=1),
                                    ResNetBlock(256, 256, stride=1),
                                    ResNetBlock(256, 256, stride=1),
                                    ResNetBlock(256, 256, stride=1))

        self.block4 = nn.Sequential(ResNetBlock(256, 512, stride=2),
                                    ResNetBlock(512, 512, stride=1),
                                    ResNetBlock(512, 512, stride=1))

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(in_features=512, out_features=output_channels)

    def forward(self, x):
        x = self.max_pool(self.relu(self.bn(self.conv(x))))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avg_pool(x)
        x = self.linear(torch.flatten(x, start_dim=1))

        return x