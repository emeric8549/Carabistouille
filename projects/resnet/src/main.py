import torch
from train import train
from models import ResNet34, Small_CNN
from utils import get_data
from visualizations import create_visualizations


dataset_name = "CIFAR10"
epochs = 100
patience = 5
batch_size = 32
lr = 1e-3
train_models = True

dataloader_train, dataloader_test, classes = get_data(batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn = Small_CNN(input_channels=3, output_channels=len(classes)).to(device)
resnet = ResNet34(input_channels=3, output_channels=len(classes)).to(device)

if __name__ == "__main__":
    if train_models:
        train(cnn, dataloader_train, dataloader_test, device, lr, epochs, patience)
        train(resnet, dataloader_train, dataloader_test, device, lr, epochs, patience)

    cnn.load_state_dict(torch.load("best_models/smallcnn.pth"))
    resnet.load_state_dict(torch.load("best_models/resnet34.pth"))
    
    create_visualizations(resnet, cnn, dataloader_test, device, classes)