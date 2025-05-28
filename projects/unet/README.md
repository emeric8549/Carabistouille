# U-Net implementation

In this project, you can find my implementation of the U-Net model.  
This model is presented in [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597) from Olaf Ronneberger, Philipp Fischer, and Thomas Brox.  
  
U-Net is a convolutional neural network architecture which has been originally designed for biomedical image segmentation. It goal is to quickly and easily find a particular subject (human, animal, car, ...).  
It follows an encoder-decoder structure with skip connections that bridge the downsampling and upsampling paths. These skip connections help preserve spatial information lost during downsampling, making U-Net particularly effective for tasks requiring precise localization.
  
### Work to do:
    - select a dataset 
    - try to train the mode
    - to compensate the pixels lost during the convolutions, must mirror the image as written in the article 
    - add image of the network