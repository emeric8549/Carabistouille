import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, nblocks, nfilters, input_shape, nclasses):
        super().__init__()
        # (3, 64, 64)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, nfilters, kernel_size=5, stride=1, padding=2),
            nn.LayerNorm([nfilters, input_shape, input_shape]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
        )

        input_shape = input_shape // 2

        # (nfilters, 32, 32)
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(nfilters, nfilters, kernel_size=5, stride=1, padding=2),
                nn.LayerNorm([nfilters, input_shape, input_shape]),
                nn.ReLU(),
            )
            for _ in range(nblocks)
        ])

        # (nfilters, 32, 32)
        self.reduce = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1), # (nfilters, 16, 16)
            nn.Conv2d(nfilters, nfilters // 2, kernel_size=5, stride=1, padding=2), # (nfilters // 2, 16, 16)
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1), # (nfilters // 2, 8, 8)
            nn.Conv2d(nfilters // 2, nfilters // 4, kernel_size=5, stride=1, padding=2),
        )

        input_shape = input_shape // 4

        # (nfilters // 4, 8, 8)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(nfilters // 4 * input_shape * input_shape, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, nclasses),
        )

    def forward(self, x):
        x = self.conv1(x)
        for block in self.conv_blocks:
            x = block(x) + x # add a residual connexion
        x = self.reduce(x)
        x = self.head(x)
        return x