from utils import *


if __name__ == "__main__":
    label_encoder = filter_images(frac_detritus=0.1, seed=42)
    resize_images()
    hist_norm_images()
    
    images_train, images_test, labels_train, labels_test = get_datasets(test_size=test_size, stratify=True, seed=seed)