import torch
from train import train
from model import ResNet34
from utils import get_data


epochs = 1000
patience = 10
batch_size = 256

dataloader_train, dataloader_test = get_data(batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = ResNet34(input_channels=3, output_channels=10).to(device)
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


best_model = train(model, dataloader_train, dataloader_test, device, lr, epochs, patience)