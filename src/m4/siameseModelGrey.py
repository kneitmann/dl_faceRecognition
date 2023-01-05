# ------------------------------- IMPORTS ------------------------------- #

import os
import matplotlib.pyplot as plt

# Tensorflow imports
import tensorflow as tf
from tensorflow import keras
import keras.backend as k
from keras.layers import Conv2D
import numpy as np
#from tensorflow.keras.applications import ResNet50


from loadDataGrey import createDataset

# ------------------------------- PARAMETERS ------------------------------- #

# Log parameters
model_name = 'siamese_model'
savedModelPath = f'./log/saved_models/{model_name}'
tb_log_dir = f'./log/tensorboard/{model_name}'
cp_filepath = f'./log/cps/{model_name}/'
training_data_path = './data/m4/training'

if not os.path.exists(cp_filepath):
    os.makedirs(cp_filepath)

cp_filepath += 'latest_weights.h5'

# Dynamic hyperparameters
learningRate = 0.001
doDataAugmentation = False
dropoutRate = 0.25
width_multiplier = 1
depth_multiplier = 1

# Training parameters
batch_size = 32
epochs = 30
validation_split = 0.2

callbacks = [
    # Checkpoint callback                    
    keras.callbacks.ModelCheckpoint(
                    filepath=cp_filepath, 
                    verbose=1, 
                    save_weights_only=True),

    # Tensorboard callback
    keras.callbacks.TensorBoard(log_dir=tb_log_dir, histogram_freq=1),

    # Early Stopping callback
    # keras.callbacks.EarlyStopping(
    #                 monitor="val_loss",
    #                 patience=2,
    #                 verbose=1)
]

# Data parameters
image_height = 160
image_width = 160

data_augmentation = keras.Sequential(
    [
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomBrightness(0.2),
    ]
)


# ----------------------------------- DATASETS ----------------------------------- #

# Creating the training and validation dataset
images_dataset, labels_dataset = createDataset(training_data_path, (image_height, image_width))


# ------------------------------- CREATING MODEL ------------------------------- #

inputs = keras.Input((64, 64, 1))
x = keras.layers.Conv2D(96, (11, 11), padding="same", activation="relu")(inputs)
x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
x = keras.layers.Dropout(0.3)(x)

x = keras.layers.Conv2D(256, (5, 5), padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
x = keras.layers.Dropout(0.3)(x)

x = keras.layers.Conv2D(384, (3, 3), padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
x = keras.layers.Dropout(0.3)(x)

pooledOutput = tf.keras.layers.GlobalAveragePooling2D()(x)
pooledOutput = tf.keras.layers.Dense(1024)(pooledOutput)
outputsCreation = tf.keras.layers.Dense(128)(pooledOutput)

modelCreation = keras.Model(inputs, outputsCreation)

# ------------------------------- CREATING MODEL INSTANCES ------------------------------- #

imgA = keras.Input(shape=(image_height, image_width, 1))
imgB = keras.Input(shape=(image_height, image_width, 1))
featA = modelCreation(imgA)
featB = modelCreation(imgB)


# ------------------------------- CREATING SIAMESE MODEL ------------------------------- #
# https://medium.com/wicds/face-recognition-using-siamese-networks-84d6f2e54ea4

def euclidean_distance(vectors):
    (featA, featB) = vectors
    sum_squared = k.sum(k.square(featA - featB), axis=1, keepdims=True)
    return k.sqrt(k.maximum(sum_squared, k.epsilon()))

distance = tf.keras.layers.Lambda(euclidean_distance)([featA, featB])
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(distance)
model = tf.keras.Model(inputs=[imgA, imgB], outputs=outputs)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])



# ------------------------------- LOAD DATA ------------------------------- #

def generate_train_image_pairs(images_dataset, labels_dataset):
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
        pos_image = images_dataset[np.random.choice(pos_indices)]
        pair_images.append((image, pos_image))
        pair_labels.append(1)

        neg_indices = np.where(labels_dataset != labels_dataset[index])
        neg_image = images_dataset[np.random.choice(neg_indices[0])]
        pair_images.append((image, neg_image))
        pair_labels.append(0)
    return np.array(pair_images), np.array(pair_labels)


# ------------------------------- TRAINING THE MODEL ------------------------------- #

images_pair, labels_pair = generate_train_image_pairs(images_dataset, labels_dataset)
history = model.fit([images_pair[:, 0], images_pair[:, 1]], labels_pair[:],validation_split=validation_split,batch_size=batch_size,epochs=epochs)


history = model.fit(
            images_pair,
            labels_pair,
            validation_split=validation_split,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks
            )


# ------------------------------- SAVING THE MODEL ------------------------------- #

model.save(savedModelPath)

def get_img_predictions(model, img_paths):
    # Loading and preprocessing the image
    img1 = tf.keras.utils.load_img(
        img_paths[0], target_size=(image_height, image_width)
    )
    img2 = tf.keras.utils.load_img(
        img_paths[1], target_size=(image_height, image_width)
    )

    img1_array = tf.keras.utils.img_to_array(img1)
    img1_array_batch = tf.expand_dims(img1_array, 0) # Create a batch

    img2_array = tf.keras.utils.img_to_array(img2)
    img2_array_batch = tf.expand_dims(img2_array, 0) # Create a batch

    # Let the model make a prediction for the image
    preds = model.predict([img1_array_batch, img2_array_batch])

    # Getting face, mask and age prediction
    pred = round(preds[0][0], 4)

    return img1, img2, pred

# Getting the image path for image to predict
img_paths = ('./data/m4/test/1_1.jpg', './data/m4/test/1_2.jpg')
img_path_split = img_paths[0].split('/')
img_name = img_path_split[len(img_path_split)-1]
img_name_split = img_name.split('_')

# Getting the actual age from the file name
if(len(img_name_split) > 1 and str.isnumeric(img_name_split[0])):
    actual = img_name_split[0]
else:
    actual = '?'

img1, img2, pred = get_img_predictions(model, img_paths)
print(f'Similarity: {pred}')

# Showing the image with the corresponding predictions
ax = plt.subplot(2, 1, 1)
plt.imshow(img1)
plt.title("Same Face: {:.2f}%)".format(pred * 100, actual))