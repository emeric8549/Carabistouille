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
import torch.nn as nn
import torch.nn.functional as F


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


def train(model, criterion, optimizer, epochs, patience, train_dataloader, test_dataloader, device):
    model = model.to(device)
    best_loss = float('inf')
    best_model = None
    counter_patience = 0

    for epoch in range(epochs):
        train_losses, train_acc = [], []
        model.train()
        print(f"Epoch {epoch + 1}")
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            outputs = torch.argmax(outputs, dim=1)
            train_acc.extend((labels == outputs).tolist())

        print(f"Avg train loss: {np.mean(train_losses):.4f}\t Avg train acc: {np.mean(train_acc):.2%}")

        test_loss, test_acc = test(model, criterion, test_dataloader, device)
        print(f"Test loss: {test_loss:.4f}\t Test acc: {test_acc:.2%}\n")

        if test_loss < best_loss:
            best_loss = test_loss
            best_model = model
            counter_patience = 0

        else:
            counter_patience += 1
            if counter_patience == patience:
                print(f"Early stopping at epoch {epoch + 1} with best loss: {best_loss:.4f}...")
                break

    return best_model

def test(model, criterion, test_dataloader, device):
    model.eval()
    loss, acc = [], []
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss.append(criterion(outputs, labels).item())
        outputs = torch.argmax(outputs, dim=1)
        acc.extend((labels == outputs).tolist())

    return np.mean(loss), np.mean(acc)


def compute_class_weights(labels):
    """
    Calculate class weights based on the frequency of each class in the dataset.
    """
    class_counts = np.bincount(labels)
    total_count = len(labels)
    class_weights = total_count / (len(class_counts) * class_counts)
    return torch.FloatTensor(class_weights)



def train_KD(teacher, student, alpha, criterion, optimizer, epochs, patience, train_dataloader, test_dataloader, device):
    teacher = teacher.to(device)
    student = student.to(device)

    best_model = None
    best_loss = float('inf')
    counter_patience = 0

    teacher.eval()

    for epoch in range(epochs):
        train_losses_total, train_losses_student, train_acc = [], [], []
        student.train()
        print(f"Epoch {epoch + 1}")
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs_student = student(inputs)

            with torch.no_grad():
                outputs_teacher = teacher(inputs)

            loss_student = criterion(outputs_student, labels)
            loss_kd = KL_softloss(outputs_student, outputs_teacher)

            loss = alpha * loss_student + (1 - alpha) * loss_kd
            loss.backward()
            optimizer.step()

            train_losses_total.append(loss.item())
            train_losses_student.append(loss_student.item())
            outputs_student = torch.argmax(outputs_student, dim=1)
            train_acc.extend((labels == outputs_student).tolist())

        print(f"Avg train loss total: {np.mean(train_losses_total):.4f}\t Avg train loss student: {np.mean(train_losses_student):.4f}\t Avg train acc: {np.mean(train_acc):.2%}")

        test_loss_total, test_loss_student, test_acc = test_KD(teacher, student, alpha, criterion, test_dataloader, device)
        print(f"Test loss total: {test_loss_total:.4f}\t Test loss student: {test_loss_student:.4f}\t Test acc: {test_acc:.2%}\n")

        if test_loss_total < best_loss:
            best_loss = test_loss_total
            best_model = student
            counter_patience = 0
        else:
            counter_patience += 1
            if counter_patience == patience:
                print(f"Early stopping at epoch {epoch + 1} with best loss: {best_loss:.4f}...")
                break

    return best_model

def test_KD(teacher, student, alpha, criterion, test_dataloader, device):
    student.eval()
    test_losses_total, test_losses_student, test_acc = [], [], []
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs_student = student(inputs)
        with torch.no_grad():
            outputs_teacher = teacher(inputs)
        
        loss_student = criterion(outputs_student, labels)
        loss_kd = KL_softloss(outputs_student, outputs_teacher)

        loss = alpha * loss_student + (1 - alpha) * loss_kd

        test_losses_total.append(loss.item())
        test_losses_student.append(loss_student.item())
        outputs_student = torch.argmax(outputs_student, dim=1)
        test_acc.extend((labels == outputs_student).tolist())

    return np.mean(test_losses_total), np.mean(test_losses_student), np.mean(test_acc)


def KL_softloss(outputs_student, outputs_teacher, temperature=4):
    outputs_student /= temperature
    outputs_teacher /= temperature

    loss = nn.KLDivLoss(reduction='batchmean')

    return loss(F.log_softmax(outputs_student, dim=1), F.softmax(outputs_teacher, dim=1)) * (temperature * temperature) 