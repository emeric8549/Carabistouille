from torch import nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self, input_channels):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x

class Decoder(nn.Module):
    def __init__(self, output_channels):
        super(Decoder, self).__init__()
        self.ConvTranspose1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.ConvTranspose2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.ConvTranspose3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.ConvTranspose4 = nn.ConvTranspose2d(in_channels=32, out_channels=output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.ConvTranspose1(x))
        x = F.relu(self.ConvTranspose2(x))
        x = F.relu(self.ConvTranspose3(x))
        x = F.tanh(self.ConvTranspose4(x))
        return x

class Autoencoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_channels)
        self.decoder = Decoder(output_channels)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x