from utils import *
from models import *

import torch
import torch.nn as nn
from torch.optim import Adam

from torch.utils.data import DataLoader
from torchvision.transforms import transforms


seed = 42
frac_detritus = 0.1
shape_images = (64, 64)
test_size = 0.2
batch_size = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    label_encoder = filter_images(frac_detritus=frac_detritus, seed=seed)
    # resize_images(shape=shape_images)
    # hist_norm_images()
    
    images_train, images_test, labels_train, labels_test = get_datasets(test_size=test_size, stratify=True, seed=seed)
    mean, std = get_stats(images_train)

    transform = transforms.Compose([
        transforms.Normalize(mean, std),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=30, translate=(0.1, 0.1))
    ])

    train_dataset = PlankthonDataset(images_train, labels_train, transform=transform)
    test_dataset = PlankthonDataset(images_test, labels_test, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)


    nblocks = 10
    nfilters = 64
    input_shape = shape_images[0]
    nclasses = len(label_encoder.classes_)

    teacher = CNN(nblocks, nfilters, input_shape, nclasses)
    student = Small_CNN(in_channels=3, out_channels=16, img_size=input_shape, nclasses=nclasses)

    epochs = 100
    patience = 10
    criterion = nn.CrossEntropyLoss()
    lr = 1e-4

    optimizer_teacher = Adam(teacher.parameters(), lr=lr)
    optimizer_student = Adam(student.parameters(), lr=lr)
    alpha = 0.5

    best_teacher = train(teacher, criterion, optimizer_teacher, epochs, patience, train_dataloader, test_dataloader, device)
    best_student = train_KD(best_teacher, student, alpha, criterion, optimizer_student, epochs, patience, train_dataloader, test_dataloader, device)