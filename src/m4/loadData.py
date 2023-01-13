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

def load_img(img_path, img_size, grayscale=False, preprocess_img=False):
    color_mode = 'rgb'
    if grayscale: color_mode = 'grayscale'

    img = keras.utils.load_img(
                img_path,
                color_mode=color_mode,
                target_size=(img_size),
                interpolation="bilinear",
                keep_aspect_ratio=True,
            )

    img = keras.utils.img_to_array(img)

    if(preprocess_img):
        img = keras.applications.mobilenet.preprocess_input(img)
    
    return img

def create_img_batch(img_array):
    img_array_batch = tf.expand_dims(img_array, 0) # Create a batch
    return img_array_batch

def createDataframe(dir):
    image_paths = []
    labels = []

    for dirpath, dirs, files in os.walk(dir): 
        for filename in files:
            image_path = os.path.join(dirpath, filename)
            label = get_label(filename)
            image_paths.append(image_path)
            labels.append(label)

    df = pd.DataFrame()
    df['image'], df['id'] = image_paths, labels

    return df

def generate_image_pairs(images_dataset, labels_dataset):
    # Making a dictionary where all the indeces of each label in the dataset is saved
    unique_labels = np.unique(labels_dataset)
    label_wise_indices = dict()
    for label in unique_labels:
        label_wise_indices.setdefault(label,
                                      [index for index, curr_label in enumerate(labels_dataset) if
                                       label == curr_label])
    
    # Creating the image pairs and labeling them
    pair_images = []
    pair_labels = []
    for index, image in enumerate(images_dataset):
        # Positive pair
        pos_indices = label_wise_indices.get(labels_dataset[index])
        rndm_pos_index = np.random.choice(pos_indices)
        pos_image = images_dataset[rndm_pos_index]
        pair_images += [[image, pos_image]]
        pair_labels += [0]

        # Negative pair
        neg_indices = np.where(labels_dataset != labels_dataset[index])
        rndm_neg_index = np.random.choice(neg_indices[0])
        neg_image = images_dataset[rndm_neg_index]
        pair_images += [[image, neg_image]]
        pair_labels += [1]

    return np.array(pair_images), np.array(pair_labels).astype("float32")

def generate_image_pairs_tf(images_dataset, labels_dataset, batch_size, shuffle=False):
    # Making a dictionary where all the indeces of each label in the dataset is saved
    unique_labels = np.unique(labels_dataset)
    label_wise_indices = dict()
    for label in unique_labels:
        label_wise_indices.setdefault(label,
                                      [index for index, curr_label in enumerate(labels_dataset) if
                                       label == curr_label])
    
    # Creating the image pairs and labeling them
    pair_images = []
    pair_labels = []
    for index, image in enumerate(images_dataset):
        # Positive pair
        pos_indices = label_wise_indices.get(labels_dataset[index])
        rndm_pos_index = np.random.choice(pos_indices)
        pos_image = images_dataset[rndm_pos_index]
        pair_images += [[image, pos_image]]
        pair_labels += [0.0]

        # Negative pair
        neg_indices = np.where(labels_dataset != labels_dataset[index])
        rndm_neg_index = np.random.choice(neg_indices[0])
        neg_image = images_dataset[rndm_neg_index]
        pair_images += [[image, neg_image]]
        pair_labels += [1.0]

    if shuffle:
        np.random.seed(42)
        np.random.shuffle(pair_images)
        np.random.seed(42)
        np.random.shuffle(pair_labels)

    images_ds = tf.data.Dataset.from_tensor_slices((np.array(pair_images)[:, 0], np.array(pair_images)[:, 1]))
    labels_ds = tf.data.Dataset.from_tensor_slices(pair_labels)

    dataset = tf.data.Dataset.zip((images_ds, labels_ds)).batch(batch_size)

    return dataset

def createDataset(dir, img_size, grayscale=False, preprocess_data=False):
    labels = []
    images = []
    image_names = []

    for dirpath, dirs, files in os.walk(dir): 
        for filename in files:
            file_path = os.path.join(dirpath, filename)
            label = get_label(filename)
            labels.append(label)

            img = load_img(file_path, img_size, grayscale, preprocess_img=preprocess_data)
            images.append(img)
            image_names.append(filename)

    assert len(images) == len(labels)        

    return np.array(images), np.array(labels)