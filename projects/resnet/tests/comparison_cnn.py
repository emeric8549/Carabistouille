from torch import nn

class small_CNN(nn.Module):
    def __init__(self, input_channels, output_features):
        super(small_CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(512)

        self.relu = nn.ReLU()

        self.max_pool = nn.MaxPool2d(kernel_size=2)

        self.linear = nn.Linear(in_features=2000, out_features=output_features)


    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        return x