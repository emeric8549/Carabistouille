from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader


def get_data(dataset_name="CIFAR10", batch_size=32, shuffle=False, download=True):
    if dataset_name == "CIFAR10":
        dataset = CIFAR10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
    elif dataset_name == "CIFAR100":    
        dataset = CIFAR100
        mean = [0.5071, 0.4865, 0.4409]
        std = [0.2673, 0.2564, 0.2761]
    else:
        raise ValueError("Dataset not recognized. Please use 'CIFAR10' or 'CIFAR100'.")


    transform = transforms.Compose([transforms.Resize(256),
                                transforms.RandomHorizontalFlip(0.5),
                                transforms.RandomCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=mean, std=std)
                                ])
    

    data_train = dataset(root='./dataset/train',
                        train=True,
                        transform=transform,
                        download=download)

    data_test = dataset(root='./dataset/test',
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

    classes = data_train.classes
    
    return dataloader_train, dataloader_test, classes