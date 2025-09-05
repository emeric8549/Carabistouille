import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
import torchvision.transforms.functional as F


def save_confusion_matrices(y, resnet_pred, cnn_pred, classes):
    true_classes = [classes[y[i]] for i in range(len(y))]
    resnet_classes = [classes[resnet_pred[i]] for i in range(len(resnet_pred))]
    cnn_classes = [classes[cnn_pred[i]] for i in range(len(cnn_pred))]

    cm_resnet = confusion_matrix(true_classes, resnet_classes, labels=classes)
    cm_cnn = confusion_matrix(true_classes, cnn_classes, labels=classes)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    disp_resnet = ConfusionMatrixDisplay(confusion_matrix=cm_resnet, display_labels=classes)
    disp_resnet.plot(ax=ax1, cmap=plt.cm.Blues, colorbar=False)
    ax1.set_title("ResNet34")

    disp_cnn = ConfusionMatrixDisplay(confusion_matrix=cm_cnn, display_labels=classes)
    disp_cnn.plot(ax=ax2, cmap=plt.cm.Blues, colorbar=False)
    ax2.set_title("Small CNN")

    plt.savefig("confusion_matrices.png")
    plt.close()
    

def save_img_predictions(imgs, true_labels, wrong_labels, classes, nrows=2, ncols=2):
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 8))

    i = 0
    for row in range(nrows):
        for col in range(ncols):
            img = imgs[i].detach()
            img = F.to_pil_image(img)

            axs[row, col].imshow(np.asarray(img))
            axs[row, col].set_title(f"resnet label: {classes[true_labels[i]]}\ncnn label: {classes[wrong_labels[i]]}")
            axs[row, col].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

            i += 1

    plt.savefig("misclassified.png")
    plt.close()


def viz_wrong(resnet, cnn, dataloader, device, classes):
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).to(device)

    imgs = None
    img_true_labels, img_cnn_labels = [], []

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        out_resnet = resnet(x)
        out_cnn = cnn(x)

        pred_resnet = torch.argmax(out_resnet, dim=1)
        pred_cnn = torch.argmax(out_cnn, dim=1)

        img_idx = torch.logical_and((pred_resnet == y), (pred_cnn != y))

        imgs = torch.cat((imgs, x[img_idx]), dim=0) if imgs is not None else x[img_idx]
        img_true_labels.extend(y[img_idx].tolist())
        img_cnn_labels.extend(pred_cnn[img_idx].tolist())

        if len(img_true_labels) >= 10:
            break

    imgs = (255 * (imgs * std + mean)).type(torch.ByteTensor)

    save_img_predictions(imgs, img_true_labels, img_cnn_labels, classes, nrows=2, ncols=3)