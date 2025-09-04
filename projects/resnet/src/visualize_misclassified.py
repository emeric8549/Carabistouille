import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F


def save(imgs, true_labels, wrong_labels, nrows=2, ncols=2):
    classes = ('plane', 'car', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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