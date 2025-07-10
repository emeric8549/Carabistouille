import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class BlurredMNIST(Dataset):
    def __init__(self, train=True, download=True, kernel=5, sigma(1, 20)):
        self.base_dataset = torchvision.datasets.MNIST(
            root='./data', train=train, download=download
        )

        self.transform_input = transforms.Compose([
            transform.GaussianBlur(kernel_size=kernel, sigma=sigma),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.transform_target = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalze((0.5,), (0.5,))
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
    dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)

    return dataloader_train, dataloader_test


class ColorizedCIFAR10(Dataset):
    def __init__(self, train=True, download=True, kernel=5, sigma(1, 20)):
        self.base_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=train, download=download
        )

        self.transform_input = transforms.Compose([
            transform.Grayscale(num_output_channels=1),
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
    dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)

    return dataloader_train, dataloader_test