import os
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset

class PlankthonDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = torch.FloatTensor(images)
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_label = self.labels[index]
        image = self.images[index]
        image = self.transform(image)
        return image, image_label



def filter_images(frac_detritus=0.1, seed=None):
    """
    Filters images from a CSV file, retaining a specified fraction of 'detritus' labeled images to balance the dataset.
    """
    # iterate over raw folder to list all images available
    idx_images = []
    for img in os.listdir('data/raw/'):
        if img.endswith('.jpg'):
            idx_images.append(img)

    df = pd.read_csv('data/raw/images.csv')
    df = df[df['image'].isin(idx_images)] # filter to only images available in the raw folder
    detritus = df[df['label'] == 'detritus']
    detritus = detritus.sample(frac=frac_detritus, random_state=seed)
    others = df[df['label'] != 'detritus']

    df_filtered = pd.concat([others, detritus])
    df_filtered.to_csv('data/filtered/filtered_images.csv', index=False)
    print(f"Original dataset size: {len(df)}")
    print(f"Filtered images saved to 'data/filtered/filtered_images.csv' with {len(df_filtered)} entries.")

def resize_images(shape=(64, 64)):
    df = pd.read_csv('data/filtered/filtered_images.csv')
    for img in df['image']:
        img_path = os.path.join('data/raw/', img)
        image = cv2.imread(img_path)
        resized_image = cv2.resize(image, shape)
        resized_path = os.path.join('data/filtered/', img)
        cv2.imwrite(resized_path, resized_image)

    print(f"Resized images saved to 'data/filtered/' with shape {shape}.")



