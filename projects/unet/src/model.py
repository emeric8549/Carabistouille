import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop

class UnetBlock(nn.Module):
    def __init__(self, input_channels, output_channels, decoder=False, downsampling=True):
        super(UnetBlock, self).__init__()

        self.decoder = decoder
        self.downsampling = downsampling

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3)
        self.upconv = nn.ConvTranspose2d(in_channels=input_channels, out_channels=output_channels, kernel_size=2) 

        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x, x_res=None):
        if self.decoder:
            x = self.upconv(x)
            x_res_cropped = CenterCrop(size=x.shape[0])(x_res)
            x = torch.cat((x_res_cropped, x), dim=1) # On concatène sur la dimension des channels

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        return x if (decoder or not downsampling) else self.max_pool(x)


class Unet(nn.Module):
    def __init__(self, input_channels):
        super(Unet, self).__init__()

        self.encoderblock1 = UnetBlock(input_channels, 64)
        self.encoderblock2 = UnetBlock(64, 128)
        self.encoderblock3 = UnetBlock(128, 256)
        self.encoderblock4 = UnetBlock(256, 512)

        self.centralblock = UnetBlock(512, 1024, downsampling=False)

        self.decoderblock1 = UnetBlock(1024, 512, decoder=True)
        self.decoderblock2 = UnetBlock(512, 256, decoder=True)
        self.decoderblock3 = UnetBlock(256, 128, decoder=True)
        self.decoderblock4 = UnetBlock(128, 64, decoder=True)

        self.conv = Conv2d(in_channels=64, out_channels=2, kernel_size=1)

    def forward(self, x):
        x1 = self.encoderblock1(x)
        x2 = self.encoderblock2(x1)
        x3 = self.encoderblock3(x2)
        x4 = self.encoderblock4(x3)

        x = self.centralblock(x4)

        x = self.decoderblock1(x, x4)
        x = self.decoderblock2(x, x3)
        x = self.decoderblock3(x, x2)
        x = self.decoderblock4(x, x1)

        return self.conv(x)