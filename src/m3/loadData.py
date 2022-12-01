import fnmatch
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

def is_number(n):
    try:
        int(n)
    except ValueError:
        return False
    return True

def createImageDataset(dir, img_size):
    labels = []
    images = []

    for dirpath, dirs, files in os.walk(dir): 
        for filename in files:
            # Loading the images
            images.append(keras.utils.img_to_array(
                            keras.utils.load_img(
                                os.path.join(dirpath, filename),
                                grayscale=False,
                                color_mode="rgb",
                                target_size=(img_size),
                                interpolation="bilinear",
                                keep_aspect_ratio=True,
                        )))

            # Creating the labels
            filename = filename.rstrip('.jpg')
            labelStrs = list(map(lambda f: int(f) if is_number(f) else f, filename.split('_'))) # If the label is a number, convert it into integer
            labels.append(labelStrs)
            
    return np.array(images), np.array(labels)