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
    filepath_split = tf.strings.split(file_path, '/')
    filename = filepath_split[len(filepath_split)-1]
    filename = tf.strings.split(filename, '.')[0]

    # If the label is a number, convert it into integer
    filename_split = tf.strings.split(filename, '_')
    face = int(filename_split[0])
    mask = int(filename_split[1])
    age = int(filename_split[2])

    return face if face >= 0 else 0, mask if mask >= 0 else 0, age if age >= 0 else 0

def load_img(img_path, img_size):
    img = tf.io.read_file(img_path)
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    return tf.image.resize(img, img_size)

def process_path(file_path):
    label = get_label(file_path)
    # Load the raw data from the file as a string
    img = load_img(file_path, (156, 156))

    return img, label

def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(32)
  ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
  return ds

def createDataset(dir, img_size):
    image_paths = []

    for dirpath, dirs, files in os.walk(dir): 
        for filename in files:
            image_paths.append(os.path.join(dirpath, filename).replace('\\', '/'))

    image_paths = tf.data.Dataset.from_tensor_slices(image_paths)
    ds = image_paths.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    
    return configure_for_performance(ds)