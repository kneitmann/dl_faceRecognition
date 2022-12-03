import fnmatch
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

def is_number(n):
    try:
        int(n)
    except ValueError:
        return False
    return True

def get_label(file_path):
    filename = file_path.rstrip('.jpg')
     # If the label is a number, convert it into integer
    label = list(map(lambda f: int(f) if is_number(f) else f, filename.split('_')))
    return label

def load_img(img_path, img_size):
    img = keras.utils.load_img(
                img_path,
                grayscale=False,
                color_mode="rgb",
                target_size=(img_size),
                interpolation="bilinear",
                keep_aspect_ratio=True,
            )
    return keras.utils.img_to_array(img)

def createDataframe(dir):
    labels = []
    image_paths = []

    for dirpath, dirs, files in os.walk(dir): 
        for filename in files:
            image_path = os.path.join(dirpath, filename)
            label = get_label(filename)
            image_paths.append(image_path)
            labels.append(label)

    np_labels = np.array(labels)
    df = pd.DataFrame()
    df['image'], df['face'], df['mask'], df['age'] = image_paths, np_labels[:, 0], np_labels[:, 1], np_labels[:, 2]

    return df

def createDataset(dir, img_size):
    labels = []
    images = []

    for dirpath, dirs, files in os.walk(dir): 
        for filename in files:
            file_path = os.path.join(dirpath, filename)
            label = get_label(filename)
            img = load_img(file_path, img_size)

            images.append(img)
            labels.append(label)

    np_labels = np.array(labels)
    np_images = np.array(images)
    return np_images, np_labels[:, 0], np_labels[:, 1], np_labels[:, 2]