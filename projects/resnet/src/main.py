import torch
from train import train
from models import ResNet34, Small_CNN
from utils import get_data
from visualize_misclassified import viz_wrong


epochs = 1000
patience = 10
batch_size = 256
lr = 1e-3
train_models = True

dataloader_train, dataloader_test = get_data(batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = ResNet34(input_channels=3, output_channels=10).to(device)
cnn = Small_CNN(input_channels=3, output_channels=10).to(device)

if __name__ == "__main__":
    if train_models:
        train(resnet, dataloader_train, dataloader_test, device, lr, epochs, patience)
        train(cnn, dataloader_train, dataloader_test, device, lr, epochs, patience)

    resnet.load_state_dict(torch.load("best_models/resnet34.pth"))
    cnn.load_state_dict(torch.load("best_models/smallcnn.pth"))

    viz_wrong(resnet, cnn, dataloader_test, device)