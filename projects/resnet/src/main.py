import argparse
import torch
from train import train
from models import ResNet34, Small_CNN
from utils import get_data
from visualizations import create_visualizations
import os

os.makedirs("best_models", exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and visualize CNN and ResNet models on CIFAR datasets.")
    parser.add_argument("--dataset", type=str, default="CIFAR10", choices=["CIFAR10", "CIFAR100"], help="Dataset to use: CIFAR10 or CIFAR100.")
    parser.add_argument("--train", type=bool, default=True, help="Flag to indicate whether to train the models.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training.")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for training.")

    args = parser.parse_args()

    dataloader_train, dataloader_test, classes = get_data(dataset_name=args.dataset, batch_size=args.batch_size, shuffle=True)

    cnn_file = "best_models/smallcnn" + args.dataset + ".pth"
    resnet_file = "best_models/resnet34" + args.dataset + ".pth"
    print(f"Using dataset: {args.dataset} with {len(classes)} classes.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = Small_CNN(input_channels=3, output_channels=len(classes)).to(device)
    resnet = ResNet34(input_channels=3, output_channels=len(classes)).to(device)

    if args.train:
        train(cnn, dataloader_train, dataloader_test, device, args.lr, args.epochs, args.patience, cnn_file)
        train(resnet, dataloader_train, dataloader_test, device, args.lr, args.epochs, args.patience, resnet_file)
    else:
        if not os.path.exists(cnn_file) or not os.path.exists(resnet_file):
            raise FileNotFoundError("Model files not found. Please train the models first.")
        
        cnn.load_state_dict(torch.load(cnn_file))
        resnet.load_state_dict(torch.load(resnet_file))


    create_visualizations(resnet, cnn, dataloader_test, device, classes, args.dataset)