def get_data_blurred(batch_size, kernel=5, sigma=(1, 20), shuffle=False, download=True):
    transform_train = transforms.Compose([
                                transforms.GaussianBlur(kernel_size=kernel, sigma=sigma),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                                ])

    transform_test = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                                ])

    X_train = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform_train)

    y_train = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform_test)

    X_test = torchvision.datasets.MNIST(root='./data', train=False,
                                            download=True, transform=transform_train)

    y_test = torchvision.datasets.MNIST(root='./data', train=False,
                                            download=True, transform=transform_test)


    dataloader_train = torch.utils.data.DataLoader((X_train, y_train), 
                                                    batch_size=batch_size, 
                                                    shuffle=shuffle,
                                                    num_workers=8)

    dataloader_test = torch.utils.data.DataLoader((X_test, y_test), 
                                                batch_size=batch_size, 
                                                shuffle=shuffle,
                                                num_workers=8)

    return dataloader_train, dataloader_test

