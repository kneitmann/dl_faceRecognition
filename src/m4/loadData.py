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

def load_img(img_path, img_size, grayscale=False):
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
    img = keras.applications.mobilenet.preprocess_input(img)
    
    return img

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
        pair_labels.append(1.0)

        neg_indices = np.where(labels_dataset != labels_dataset[index])
        rndm_neg_index = np.random.choice(neg_indices[0])
        neg_image = images_dataset[rndm_neg_index]
        pair_images.append((image, neg_image))
        pair_labels.append(0.0)

    return np.array(pair_images), np.array(pair_labels)
    
def generate_image_pairs_alt(images_dataset, labels_dataset, min_equals = 1000):
    pairs = []
    labels = []
    equal_items = 0
    
    #index with all the positions containing a same value
    # Index[1] all the positions with values equals to 1
    # Index[2] all the positions with values equals to 2
    #.....
    # Index[9] all the positions with values equals to 9 
    index = [np.where(labels_dataset == i)[0] for i in range(10)]
    
    for n_item in range(len(images_dataset)): 
        if equal_items < min_equals:
            #Select the number to pair from index containing equal values. 
            num_rnd = np.random.randint(len(index[labels_dataset[n_item]]))
            num_item_pair = index[labels_dataset[n_item]][num_rnd]

            equal_items += 1
        else: 
            #Select any number in the list 
            num_item_pair = np.random.randint(len(images_dataset))
            
        #I'm not checking that numbers is different. 
        #That's why I calculate the label depending if values are equal. 
        labels += [int(labels_dataset[n_item] == labels_dataset[num_item_pair])]         
        pairs += [[images_dataset[n_item], images_dataset[num_item_pair]]]
            
    return np.array(pairs), np.array(labels).astype('float32') 

def createDataset(dir, img_size, grayscale=False):
    labels = []
    images = []
    image_names = []

    for dirpath, dirs, files in os.walk(dir): 
        for filename in files:
            file_path = os.path.join(dirpath, filename)
            label = get_label(filename)
            labels.append(label)

            img = load_img(file_path, img_size, grayscale)
            images.append(img)
            image_names.append(filename)

    assert len(images) == len(labels)

    # Shuffling the data
    np.random.seed(123)
    np.random.shuffle(images)
    np.random.seed(123)
    np.random.shuffle(labels)

    return np.array(images), np.array(labels)