from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class MNIST_AE(Dataset):
    def __init__(self, train=True, download=True):
        self.base_dataset = torchvision.datasets.MNIST(
            root='./data', train=train, download=download
        )

        self.transform_input = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.transform_target = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, _ = self.base_dataset[idx]
        input_img = self.transform_input(img)
        target_img = self.transform_target(img)
        return input_img, target_img



def get_mnist(batch_size, shuffle=False, download=True):
    train_dataset = MNIST_AE(train=True, download=download)
    test_dataset = MNIST_AE(train=False, download=download)

    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
    dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)

    return dataloader_train, dataloader_test



class BlurredMNIST(Dataset):
    def __init__(self, train=True, download=True, kernel=5, sigma=(1, 20)):
        self.base_dataset = torchvision.datasets.MNIST(
            root='./data', train=train, download=download
        )

        self.transform_input = transforms.Compose([
            transforms.GaussianBlur(kernel_size=kernel, sigma=sigma),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.transform_target = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, _ = self.base_dataset[idx]
        input_img = self.transform_input(img)
        target_img = self.transform_target(img)
        return input_img, target_img


        
def get_data_blurred(batch_size, kernel=5, sigma=(1, 20), shuffle=False, download=True):
    train_dataset = BlurredMNIST(train=True, download=download, kernel=kernel, sigma=sigma)
    test_dataset = BlurredMNIST(train=False, download=download, kernel=kernel, sigma=sigma)

    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
    dataloader_test = DataLoader(test_dataset, batch_size=4, shuffle=shuffle, num_workers=8)

    return dataloader_train, dataloader_test


class ColorizedCIFAR10(Dataset):
    def __init__(self, train=True, download=True):
        self.base_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=train, download=download
        )

        self.transform_input = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.transform_target = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, _ = self.base_dataset[idx]
        input_img = self.transform_input(img)
        target_img = self.transform_target(img)
        return input_img, target_img


        
def get_data_colorized(batch_size, shuffle=False, download=True):
    train_dataset = ColorizedCIFAR10(train=True, download=download)
    test_dataset = ColorizedCIFAR10(train=False, download=download)

    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
    dataloader_test = DataLoader(test_dataset, batch_size=4, shuffle=shuffle, num_workers=8)

    return dataloader_train, dataloader_test



def imcompare(X, pred, real):
    X = X / 2 + 0.5
    pred = pred / 2 + 0.5
    real = real / 2 + 0.5

    X_np = torchvision.utils.make_grid(X, nrow=len(X)).numpy()
    pred_np = torchvision.utils.make_grid(pred, nrow=len(pred)).numpy()
    real_np = torchvision.utils.make_grid(real, nrow=len(real)).numpy()

    fig, axes = plt.subplots(3, 1, figsize=(12, 6))
    axes[0].imshow(np.transpose(X_np, (1, 2, 0)))
    axes[0].set_title("Blurred images")
    axes[0].axis("off")

    axes[1].imshow(np.transpose(pred_np, (1, 2, 0)))
    axes[1].set_title("Model outputs")
    axes[1].axis("off")

    axes[2].imshow(np.transpose(real_np, (1, 2, 0)))
    axes[2].set_title("Real images")
    axes[2].axis("off")

    plt.tight_layout()
    current_datetime = datetime.now()
    path = "model_results/" + current_datetime.strftime("%Y-%m-%d %H_%M_%S")
    plt.savefig(path)

    print(f"Results saved in {path}")