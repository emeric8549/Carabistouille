import torch
from train import train
from models import ResNet34, Small_CNN
from utils import get_data


epochs = 1000
patience = 10
batch_size = 256
lr = 1e-3

dataloader_train, dataloader_test = get_data(batch_size=batch_size, shuffle=True)

model = ResNet34(input_channels=3, output_channels=10).to(device) # or Small_CNN
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


best_model = train(model, dataloader_train, dataloader_test, device, lr, epochs, patience)