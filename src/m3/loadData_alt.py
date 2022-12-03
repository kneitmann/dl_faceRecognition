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
    #label = list(map(lambda f: int(f) if is_number(f) else f, filename.split('_')))
    filename_split = filename.split('_')
    #return label
    return int(filename_split[0]), int(filename_split[1]), int(filename_split[2])

def load_img(img_path, img_size):
    img = tf.io.read_file(img_path)
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    return tf.image.resize(img, img_size)

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
    #labels = []
    face_labels = []
    mask_labels = []
    age_labels = []
    images = []

    for dirpath, dirs, files in os.walk(dir): 
        for filename in files:
            file_path = os.path.join(dirpath, filename)
            #label = get_label(filename)
            face, mask, age = get_label(filename)
            face_labels.append(face)
            mask_labels.append(mask)
            age_labels.append(age)

            img = load_img(file_path, img_size)
            images.append(img)
            #labels.append(label)

    #np_labels = np.array(labels)
    #np_images = np.array(images)
    #return np_images, np_labels[:, 0], np_labels[:, 1], np_labels[:, 2]
    return tf.convert_to_tensor(images), tf.convert_to_tensor(face_labels), tf.convert_to_tensor(mask_labels), tf.convert_to_tensor(age_labels)