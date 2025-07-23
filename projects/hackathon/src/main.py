from utils import *
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

seed = 42
frac_detritus = 0.1
test_size = 0.2
batch_size = 256

if __name__ == "__main__":
    label_encoder = filter_images(frac_detritus=frac_detritus, seed=seed)
    resize_images()
    hist_norm_images()
    
    images_train, images_test, labels_train, labels_test = get_datasets(test_size=test_size, stratify=True, seed=seed)
    mean, std = get_stats(images_train)

    transform = transforms.Compose([
        transforms.Normalize(mean, std),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=30, translate(0.1, 0.1))
    ])

    train_dataset = PlankthonDataset(images_train, labels_train, transform=transform)
    test_dataset = PlankthonDataset(images_test, labels_test, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)