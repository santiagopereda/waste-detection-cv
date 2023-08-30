import pandas as pd
import numpy as np
import base64
import os
from PIL import Image
import matplotlib.pyplot as plt # used this in jupyter to show the image (plt.imshow)

IMAGE_PATH  = 'preprocessing/images' #This is the path to the images folder where the user
                         # image is being dropped

def reshape_img(image_path):
    '''
    Input a the path of the image and output its resized
    version (640x640)
    '''
    #Getting the image from the ./images directory
    im = os.listdir(image_path)[0]

    #Opening the image using the .open method from PIL library
    img = Image.open(image_path+'/'+im)

    # resizing the image to 640x640
    img_resized = img.resize((640,640))
    print('Image resized✅')
    return img_resized


if __name__ == '__main__':
   print(plt.imshow(reshape_img(IMAGE_PATH)))
