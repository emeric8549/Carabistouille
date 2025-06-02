import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop

class UNetBlock(nn.Module):
    def __init__(self, input_channels, output_channels, decoder=False, downsampling=True):
        super(UNetBlock, self).__init__()

        self.decoder = decoder
        self.downsampling = downsampling

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3)
        self.upconv = nn.ConvTranspose2d(in_channels=input_channels, out_channels=output_channels, kernel_size=2, stride=2) 

        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x, x_res=None):
        if self.decoder:
            x = self.upconv(x)
            x_res_cropped = CenterCrop(size=x.shape[2])(x_res)
            x = torch.cat((x_res_cropped, x), dim=1) # On concatène sur la dimension des channels

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        return x if (self.decoder or not self.downsampling) else self.max_pool(x), x


class UNet(nn.Module):
    def __init__(self, input_channels):
        super(UNet, self).__init__()

        self.encoderblock1 = UNetBlock(input_channels, 64)
        self.encoderblock2 = UNetBlock(64, 128)
        self.encoderblock3 = UNetBlock(128, 256)
        self.encoderblock4 = UNetBlock(256, 512)

        self.centralblock = UNetBlock(512, 1024, downsampling=False)

        self.decoderblock1 = UNetBlock(1024, 512, decoder=True)
        self.decoderblock2 = UNetBlock(512, 256, decoder=True)
        self.decoderblock3 = UNetBlock(256, 128, decoder=True)
        self.decoderblock4 = UNetBlock(128, 64, decoder=True)

        self.conv = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)

    def forward(self, x):
        x, x1 = self.encoderblock1(x)
        x, x2 = self.encoderblock2(x)
        x, x3 = self.encoderblock3(x)
        x, x4 = self.encoderblock4(x)
        
        x, _ = self.centralblock(x)

        x, _ = self.decoderblock1(x, x4)
        x, _ = self.decoderblock2(x, x3)
        x, _ = self.decoderblock3(x, x2)
        x, _ = self.decoderblock4(x, x1)

        return self.conv(x)