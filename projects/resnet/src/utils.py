from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader


def get_data(batch_size, shuffle=False, download=True):
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.RandomHorizontalFlip(0.5),
                                    transforms.RandomCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # We use the default per_pixel normalization based on the ImageNet dataset
                                    ])

    data_train = CIFAR10(root='./dataset/train',
                        train=True,
                        transform=transform,
                        download=download)

    data_test = CIFAR10(root='./dataset/test',
                        train=False,
                        transform=transform,
                        download=download)

    dataloader_train = DataLoader(data_train,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=8)

    dataloader_test = DataLoader(data_test,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=8)

    return dataloader_train, dataloader_test