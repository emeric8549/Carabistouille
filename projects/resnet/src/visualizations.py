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
    disp_resnet.plot(ax=ax1, cmap=plt.cm.cividis, colorbar=False)
    ax1.set_title("ResNet34")

    disp_cnn = ConfusionMatrixDisplay(confusion_matrix=cm_cnn, display_labels=classes)
    disp_cnn.plot(ax=ax2, cmap=plt.cm.cividis, colorbar=False)
    ax2.set_title("Small CNN")

    plt.savefig("confusion_matrices.png")
    plt.close()


def save_img_predictions(imgs, true_labels, wrong_labels, idx, classes, nrows=2, ncols=2):
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 8))

    i = 0
    for row in range(nrows):
        for col in range(ncols):
            img = imgs[i].detach()
            img = F.to_pil_image(img)

            axs[row, col].imshow(np.asarray(img))
            axs[row, col].set_title(f"resnet label: {classes[true_labels[idx[i]]]}\ncnn label: {classes[wrong_labels[idx[i]]]}")
            axs[row, col].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

            i += 1

    plt.savefig("misclassified.png")
    plt.close()


def create_visualizations(resnet, cnn, dataloader, device, classes, dataset_name):
    if dataset_name == "CIFAR10":
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1).to(device)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).reshape(1, 3, 1, 1).to(device)
    else:
        mean = torch.tensor([0.5071, 0.4865, 0.4409]).reshape(1, 3, 1, 1).to(device)
        std = torch.tensor([0.2673, 0.2564, 0.2761]).reshape(1, 3, 1, 1).to(device)

    imgs = None
    img_idx_list, y_true_list, resnet_pred_list, cnn_pred_list = [], [], [], []

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        out_resnet = resnet(x)
        out_cnn = cnn(x)

        pred_resnet = torch.argmax(out_resnet, dim=1)
        pred_cnn = torch.argmax(out_cnn, dim=1)

        img_idx = torch.logical_and((pred_resnet == y), (pred_cnn != y)).tolist()

        if imgs is None or imgs.shape[0] < 10:
            img_idx_list.extend(img_idx)
            imgs = torch.cat((imgs, x[img_idx]), dim=0) if imgs is not None else x[img_idx]

        y_true_list.extend(y.tolist())
        resnet_pred_list.extend(pred_resnet.tolist())
        cnn_pred_list.extend(pred_cnn.tolist())

    idx = np.where(np.array(img_idx_list) == 1)[0]

    imgs_unnormalized = (255 * (imgs * std + mean)).type(torch.ByteTensor)
    save_confusion_matrices(y_true_list, resnet_pred_list, cnn_pred_list, classes)
    save_img_predictions(imgs_unnormalized, y_true_list, cnn_pred_list, idx, classes, nrows=2, ncols=3)