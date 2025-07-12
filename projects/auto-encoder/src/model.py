from torch import nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self, input_channels):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

class Decoder(nn.Module):
    def __init__(self, output_channels):
        super(Decoder, self).__init__()
        self.ConvTranspose1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.ConvTranspose2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.ConvTranspose3 = nn.ConvTranspose2d(in_channels=64, out_channels=output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.ConvTranspose1(x))
        x = F.relu(self.ConvTranspose2(x))
        x = F.tanh(self.ConvTranspose3(x))
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



class Encoder_SC(nn.Module):
    def __init__(self, input_channels):
        super(Encoder_SC, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(512)

    def forward(self, x):
        skip_connection_1 = F.relu(self.bn1(self.conv1(x)))
        skip_connection_2 = F.relu(self.bn2(self.conv2(skip_connection_1)))
        x = F.relu(self.bn3(self.conv3(skip_connection_2)))
        return x, [skip_connection_1, skip_connection_2]

class Decoder_SC(nn.Module):
    def __init__(self, output_channels):
        super(Decoder_SC, self).__init__()
        self.ConvTranspose1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.ConvTranspose2 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.ConvTranspose3 = nn.ConvTranspose2d(in_channels=256, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)

    def forward(self, x, skips):
        skip_connection_1, skip_connection_2 = skips
        x = F.relu(self.bn1(self.ConvTranspose1(x)))
        x = torch.cat((x, skip_connection_2), dim=1)
        x = F.relu(self.bn2(self.ConvTranspose2(x)))
        x = torch.cat((x, skip_connection_1), dim=1)
        x = F.tanh(self.ConvTranspose3(x))
        return x

class Autoencoder_SC(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Autoencoder_SC, self).__init__()
        self.encoder = Encoder_SC(input_channels)
        self.decoder = Decoder_SC(output_channels)

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.decoder(x, skips)
        return x