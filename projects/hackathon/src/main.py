from utils import filter_images, resize_images, hist_norm_images


if __name__ == "__main__":
    filter_images(frac_detritus=0.1, seed=42)
    resize_images()
    hist_norm_images()