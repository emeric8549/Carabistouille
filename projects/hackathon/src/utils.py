import os
import pandas as pd
import numpy as np
from tqdm import tqdm

import cv2
from skimage import io, exposure, img_as_ubyte

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import Dataset



class PlankthonDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
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

    label_encoder = LabelEncoder()
    labels = df_filtered['label'].values.tolist()
    label_encoder.fit(labels)
    df_filtered['class'] = df_filtered.apply(lambda row: label_encoder.transform([row['label']])[0], axis=1)

    df_filtered.to_csv('data/filtered_images.csv', index=False)
    print(f"Original dataset size: {len(df)}")
    print(f"Filtered images saved to 'data/filtered_images.csv' with {len(df_filtered)} entries.")
    print(f"There are {len(label_encoder.classes_)} classes in this dataset")

    return label_encoder


def resize_images(shape=(64, 64)):
    df = pd.read_csv('data/filtered_images.csv')
    for img in tqdm(df['image'], desc=f"Resizing images with shape {shape}"):
        img_path = os.path.join('data/raw/', img)
        image = cv2.imread(img_path)
        resized_image = cv2.resize(image, shape)
        resized_path = os.path.join('data/resized/', img)
        cv2.imwrite(resized_path, resized_image)


def hist_norm_images(clip_limit=0.02, kernel_size=8):
    for img in tqdm(os.listdir('data/resized/'), desc="Histogram normalization"):
        image_path = os.path.join('data/resized/', img)
        image = io.imread(image_path)
        img_eq = img_as_ubyte(exposure.equalize_adapthist(image, clip_limit=clip_limit, kernel_size=kernel_size))
        new_image_path = os.path.join('data/norm/', img)
        io.imsave(new_image_path, img_eq)


def get_datasets(test_size=0.2, stratify=False, seed=None):
    images, labels = [], []
    df = pd.read_csv('data/filtered_images.csv')
    for _, row in df.iterrows():
        image = cv2.imread(os.path.join('data/norm/', row['image']))
        images.append(image / 255)
        labels.append(row['class'])

    images = np.transpose(np.stack(images), axes=(0, 3, 1, 2))
    images_train, images_test, labels_train, labels_test = train_test_split(
                                                                images, 
                                                                labels, 
                                                                test_size=test_size, 
                                                                stratify=labels if stratify else None, 
                                                                random_state=seed
                                                                )

    return torch.FloatTensor(images_train), torch.FloatTensor(images_test), torch.LongTensor(labels_train), torch.LongTensor(labels_test)


def get_stats(dataset_train):
    mean = torch.mean(dataset_train, dim=(0, 2, 3))
    std = torch.std(dataset_train, dim=(0, 2, 3))

    return mean, std