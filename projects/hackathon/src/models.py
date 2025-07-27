import torch
import torch.nn as nn
from einops import repeat
from einops.layers.torch import Rearrange, Reduce

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



class PatchEmbedding(nn.Module):
    def __init__(self, out_channels, emb_size, in_channels=3, patch_size=8, img_size=64):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size)
        self.patching = Rearrange('b c h w -> b (h w) c') # Flatten h and w
        self.projection = nn.Linear(out_channels, emb_size)

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size)) # at the end of encoder, this token should contain information about all the patches
        self.positional_embedding = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.projection(x)
        x = self.patching(x)
        x = self.projection(x)

        cls_token = repeat(self.cls_token, '() n e -> b n e', b=batch_size)
        x = torch.cat([cls_token, x], dim=1)
        x += self.positional_embedding

        return x


class ClassificationHead(nn.Module):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.cls_proj = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes),
        )

    def forward(self, x):
        cls_pred = self.cls_proj(x[:, :-1])
        return cls_pred


class VisionTransformer(nn.Module):
    def __init__(self, emb_size, dim_feedforward, n_head, n_layers, dropout=0, in_channels=3, out_channels=32, patch_size=8, img_size=64, n_classes=84):
        super().__init__()
        self.patch_embedding = PatchEmbedding(out_channels, emb_size, in_channels, patch_size, img_size)

        encoder_layer = nn.TransformerEncoderLayer(emb_size, n_head, dim_feedforward, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

        self.head = ClassificationHead(emb_size, n_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer(x)
        x = self.head(x)
        return x