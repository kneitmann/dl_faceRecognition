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

def get_label(file_path, for_regression=False, use_age_groups=False):
    filename = file_path.rstrip('.jpg')
     # If the label is a number, convert it into integer
    filename_split = filename.split('_')
    face = int(filename_split[0])
    mask = int(filename_split[1])
    age = int(filename_split[2])
    
    if use_age_groups:
        if age >= 0 & age <= 13:
            age = 0
        elif age > 13 & age <= 19:
            age = 1
        elif age > 19 & age <= 29:
            age = 2
        elif age > 29 & age <= 39:
            age = 3
        elif age > 39 & age <= 49:
            age = 4
        elif age > 49 & age <= 59:
            age = 5
        elif age > 59:
            age = 6
        else:
            age = 7
    else:
        if age < 0 or age > 120:
            age = 121

    if not for_regression:
        # face_onehot = np.zeros(2)
        # face_onehot[face] = 1.0 # One-hot array
        # face = face_onehot

        # mask_onehot = np.zeros(2)
        # mask_onehot[mask] = 1.0 # One-hot array
        # mask = mask_onehot
        
        age_onehot = np.zeros(122)
        age_onehot[age] = 1.0 # One-hot array
        age = age_onehot
    
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

def load_images(dir):
    return

def createDataset(dir, img_size, for_regression=False, use_age_groups=False):
    face_labels = []
    mask_labels = []
    age_labels = []
    images = []

    for dirpath, dirs, files in os.walk(dir): 
        for filename in files:
            file_path = os.path.join(dirpath, filename)
            face, mask, age = get_label(filename, for_regression, use_age_groups)
            face_labels.append(face)
            mask_labels.append(mask)
            age_labels.append(age)

            img = load_img(file_path, img_size)
            images.append(img)

    return np.array(images), np.array(face_labels), np.array(mask_labels), np.array(age_labels)