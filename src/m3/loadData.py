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
    filename_split = filename.split('_')
    face = int(filename_split[0])
    mask = int(filename_split[1])
    age = int(filename_split[2])

    return face, mask, age

def load_img(img_path, img_size):
    img = keras.utils.load_img(
                img_path,
                grayscale=False,
                color_mode="rgb",
                target_size=(img_size),
                interpolation="bilinear",
                keep_aspect_ratio=True,
            )

    img = keras.utils.img_to_array(img)
    img = keras.applications.mobilenet.preprocess_input(img)
    
    return img

def createDataframe(dir):
    image_paths = []
    face_labels = []
    mask_labels = []
    age_labels = []

    for dirpath, dirs, files in os.walk(dir): 
        for filename in files:
            image_path = os.path.join(dirpath, filename)
            face, mask, age = get_label(filename)
            image_paths.append(image_path)
            face_labels.append(face)
            face_labels.append(mask)
            face_labels.append(age)

    df = pd.DataFrame()
    df['image'], df['face'], df['mask'], df['age'] = image_paths, face_labels, mask_labels, age_labels

    return df

def createDataset(dir, img_size):
    face_labels = []
    mask_labels = []
    age_labels = []
    images = []

    for dirpath, dirs, files in os.walk(dir): 
        for filename in files:
            file_path = os.path.join(dirpath, filename)
            face, mask, age = get_label(filename)
            face_labels.append(face)
            mask_labels.append(mask)
            age_labels.append(age)

            img = load_img(file_path, img_size)
            images.append(img)

    #assert len(face_labels) == len(mask_labels) == len(age_labels) == len(images)
    #p = np.random.permutation(len(face_labels))

    return np.array(images), np.array(face_labels), np.array(mask_labels), np.array(age_labels)