import fnmatch
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import math

def is_number(n):
    try:
        int(n)
    except ValueError:
        return False
    return True

def get_label(file_path, for_regression=False, use_age_groups=False):
    filepath_split = tf.strings.split(file_path, '/')
    filename = filepath_split[len(filepath_split)-1]
    filename = tf.strings.split(filename, '.')[0]

    # If the label is a number, convert it into integer
    filename_split = tf.strings.split(filename, '_')
    face = int(filename_split[0])
    mask = int(filename_split[1])
    age = int(filename_split[2])

    if(use_age_groups):
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

        if not for_regression:
            # face = tf.one_hot(face, 2)
            # mask = tf.one_hot(mask, 2)
            age = tf.one_hot(age, 8)
    else:
        if age < 0 or age > 120:
            age = 121

        if not for_regression:
            # face = tf.one_hot(face, 2)
            # mask = tf.one_hot(mask, 2)
            age = tf.one_hot(age, 122)

    return face, mask, age

def load_img(img_path, img_size):
    img = tf.io.read_file(img_path)
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    img = tf.image.resize(img, img_size)
    # Preprocess image
    img = keras.applications.resnet.preprocess_input(img)

    return img

def process_path(file_path, img_size, for_regression=False, use_age_groups=False):
    label = get_label(file_path, for_regression, use_age_groups)
    # Load the raw data from the file as a string
    img = load_img(file_path, img_size)

    return img, label

def configure_for_performance(ds, batch_size):
  ds = ds.cache()
  ds = ds.shuffle(10000, seed=123)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
  return ds

def createDataset(dir, img_size, batch_size, validation_split, for_regression=False, use_age_groups=False):
    image_paths = []

    for dirpath, dirs, files in os.walk(dir): 
        for filename in files:
            image_paths.append(os.path.join(dirpath, filename).replace('\\', '/'))

    val_data_size = int(len(image_paths) * validation_split)

    image_paths = tf.data.Dataset.from_tensor_slices(image_paths)
    train_paths = image_paths.skip(val_data_size)
    val_paths = image_paths.take(val_data_size)

    process_path_fn = lambda path: process_path(path, img_size, for_regression, use_age_groups)
    train_ds = train_paths.map(process_path_fn, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_paths.map(process_path_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    return configure_for_performance(train_ds, batch_size), configure_for_performance(val_ds, batch_size)