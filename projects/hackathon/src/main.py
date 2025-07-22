from utils import *


if __name__ == "__main__":
    label_encoder = filter_images(frac_detritus=0.1, seed=42)
    resize_images()
    hist_norm_images()
    convert_to_npy()