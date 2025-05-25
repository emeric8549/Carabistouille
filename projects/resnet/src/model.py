import torch.nn as nn

class ResNet(nn.module):
    def __init__(self, input_channels, hidden_dims, output_dim):
        super(ResNet, self).__init__()

        # Créer des blocs de 2 couches pour simplifier ?
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=7, stride=2) # Probablement modifier padding et/ou stride pour diviser par deux la dimension de l'image

        self.max_pool = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same')
        
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same')
        self.conv9 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same')
        self.conv10 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same')
        self.conv11 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same')
        self.conv12 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same')
        self.conv13 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same')
        self.conv14 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same')

        self.conv15 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2)
        self.conv16 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same')
        self.conv17 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same')
        self.conv18 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same')
        self.conv19 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same')
        self.conv20 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same')
        self.conv21 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same')
        self.conv22 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same')
        self.conv23 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same')
        self.conv24 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same')
        self.conv25 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same')
        self.conv26 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same')
        
        self.conv27 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2)
        self.conv28 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same')
        self.conv29 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same')
        self.conv30 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same')
        self.conv31 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same')
        self.conv32 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same')

        self.avg_pool = nn.AvgPool2d(kernel_size=3)

        self.linear = nn.Linear(in_features= , out_features=1000)


    def forward(self, x):
        