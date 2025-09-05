import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as F


def save(imgs, true_labels, wrong_labels, classes, nrows=2, ncols=2):
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

    save(imgs, img_true_labels, img_cnn_labels, classes, nrows=2, ncols=3)