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
    ds = keras.utils.image_dataset_from_directory(
        "../../data/m3/training/",
        label_mode=None,
        color_mode='rgb',
        batch_size=32,
        image_size=img_size,
        seed=123,
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=True,
    )

    for batch in ds:
        for data in batch:
            images.append(np.array(data))

    for dirpath, dirs, files in os.walk(dir): 
        for filename in files:
            # Loading the images
            # img = keras.utils.load_img(
            #                     os.path.join(dirpath, filename),
            #                     grayscale=False,
            #                     color_mode="rgb",
            #                     target_size=(img_size),
            #                     interpolation="bilinear",
            #                     keep_aspect_ratio=True,
            #             )
            #images.append(keras.utils.img_to_array(img))

            # Creating the labels
            filename = filename.rstrip('.jpg')
            labelStrs = list(map(lambda f: int(f) if is_number(f) else f, filename.split('_'))) # If the label is a number, convert it into integer
            labels.append(labelStrs)

    return np.array(images), np.array(labels)