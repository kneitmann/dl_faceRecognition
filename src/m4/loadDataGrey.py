import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

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
    id = int(filename_split[0])
    
    return id

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

def createDataframe(dir, for_regression=False):
    image_paths = []
    labels = []

    for dirpath, dirs, files in os.walk(dir): 
        for filename in files:
            image_path = os.path.join(dirpath, filename)
            label = get_label(filename, for_regression)
            image_paths.append(image_path)
            labels.append(label)

    df = pd.DataFrame()
    df['image'], df['id'] = image_paths, labels

    return df

def generate_image_pairs(image_names, images_dataset, labels_dataset):
    unique_labels = np.unique(labels_dataset)
    label_wise_indices = dict()
    for label in unique_labels:
        label_wise_indices.setdefault(label,
                                      [index for index, curr_label in enumerate(labels_dataset) if
                                       label == curr_label])
    
    pair_images = []
    pair_labels = []
    for index, image in enumerate(images_dataset):
        pos_indices = label_wise_indices.get(labels_dataset[index])
        rndm_pos_index = np.random.choice(pos_indices)
        pos_image = images_dataset[rndm_pos_index]
        pair_images.append((image, pos_image))
        pair_labels.append(1)
        print(f'Positive image pair: {image_names[index]}, {image_names[rndm_pos_index]}')

        neg_indices = np.where(labels_dataset != labels_dataset[index])
        rndm_neg_index = np.random.choice(neg_indices[0])
        neg_image = images_dataset[rndm_neg_index]
        pair_images.append((image, neg_image))
        pair_labels.append(0)
        print(f'Negative image pair: {image_names[index]}, {image_names[rndm_neg_index]}')

    return np.array(pair_images), np.array(pair_labels)

def createDataset(dir, img_size):
    labels = []
    images = []
    image_names = []

    for dirpath, dirs, files in os.walk(dir): 
        for filename in files:
            file_path = os.path.join(dirpath, filename)
            label = get_label(filename)
            labels.append(label)

            img = load_img(file_path, img_size)
            images.append(img)
            image_names.append(filename)

    assert len(images) == len(labels)

    np.random.seed(42)
    np.random.shuffle(images)
    np.random.seed(42)
    np.random.shuffle(labels)
    np.random.seed(42)
    np.random.shuffle(image_names)

    return generate_image_pairs(image_names, np.array(images), np.array(labels))