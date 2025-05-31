from torch import nn


class small_CNN(nn.Module):
    def __init__(self, input_channels, output_features):
        super(small_CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(512)

        self.relu = nn.ReLU()

        self.max_pool = nn.MaxPool2d(kernel_size=2)

        self.linear = nn.Linear(in_features=2000, out_features=output_features)


    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        return x


from torchvision.datasets import CIFAR10
from torchvision import transforms


transform = transforms.Compose([transforms.Resize(256),
                                transforms.RandomHorizontalFlip(0.5),
                                transforms.RandomCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # We use the default per_pixel normalization based on the ImageNet dataset
                                ])


data_train = CIFAR10(root='./dataset/train',
                     train=True,
                     transform=transform,
                     download=True)

data_test = CIFAR10(root='./dataset/test',
                     train=False,
                     transform=transform,
                     download=True)

dataloader_train = DataLoader(data_train,
                              batch_size=256,
                              shuffle=True,
                              num_workers=8)

dataloader_test = DataLoader(data_test,
                              batch_size=256,
                              shuffle=True,
                              num_workers=8)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    cnn = small_CNN(input_channels=3, output_features=10).to(device)
    print(f"Number of parameters: {sum(p.numel() for p in cnn.parameters() if p.requires_grad)}")
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(cnn.parameters(), weight_decay=1e-4, momentum=0.9, lr=0.1)
    
    for epoch in range(100):
        losses = []
        for x, y in tqdm(dataloader_train, desc="Training"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            pred = cnn(x)
            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if (epoch+1) % 1 == 0:
            losses_test = []
            for x, y in tqdm(dataloader_test, desc="Testing"):
                x, y = x.to(device), y.to(device)
                pred = cnn(x)
                losses_test.append(criterion(pred, y).item())

            print(f"Epoch {epoch+1} | Train loss: {np.mean(np.array(losses)):4f} | Test loss: {np.mean(np.array(losses_test))}")