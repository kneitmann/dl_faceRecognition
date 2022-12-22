# ------------------------------- IMPORTS ------------------------------- #

import os

# Tensorflow imports
import tensorflow as tf
from tensorflow import keras
import keras.backend as k
from tensorflow.keras.applications import ResNet50

from loadData import createDataset

# ------------------------------- PARAMETERS ------------------------------- #

# Log parameters
model_name = 'siamese_model'
savedModelPath = f'../../log/saved_models/{model_name}'
tb_log_dir = f'../../log/tensorboard/{model_name}'
cp_filepath = f'../../log/cps/{model_name}/'

if not os.path.exists(cp_filepath):
    os.makedirs(cp_filepath)

cp_filepath += 'latest_weights.h5'

# Dynamic hyperparameters
learningRate = 0.001
doDataAugmentation = True
dropoutRate = 0.25
width_multiplier = 1
depth_multiplier = 1

# Training parameters
batch_size = 32
epochs = 10
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
train_image_pairs, train_labels = createDataset('../../data/m3/training', (image_height, image_width))

# ------------------------------- CREATING MODEL ------------------------------- #

# Loading either the MobileNet architecture model or the previously saved model, and freeze it for transfer learning
base = ResNet50(
                input_shape=(image_height, image_width, 3), # Optional shape tuple, only to be specified if include_top is False
                # alpha=width_multiplier, # Controls the width of the network. (Width multiplier)
                # depth_multiplier=depth_multiplier, # Depth multiplier for depthwise convolution. (Resolution multiplier)
                # dropout=dropoutRate, # Dropout rate. Default to 0.001.
                weights="imagenet",
                # input_tensor=None,
                # pooling='avg', # Optional pooling mode for feature extraction when include_top is False. (None, avg, max)
                include_top=False
                )
           
# Freeze the base model
base.trainable = False

inputs = keras.Input(shape=(image_height, image_width, 3))

# Data Augmentation on input
if(doDataAugmentation):
    inputs = data_augmentation(inputs)

# Running base model in inference mode
base_model = base(inputs, training=False)
base_model = keras.layers.GlobalAveragePooling2D()(base_model)
base_model = keras.layers.Dense(1024)(base_model)
base_model = keras.layers.Dropout(0.3)(base_model)

# Add Dense layer
top_model = tf.keras.layers.Dense(256, activation='relu')(base_model)
top_model = keras.layers.BatchNormalization()(top_model)
top_model = keras.layers.Activation('relu')(top_model)
top_model = keras.layers.Dropout(0.2)(top_model)

outputs = tf.keras.layers.Dense(128, activation='relu')(top_model)


# ------------------------------- CREATING MODEL INSTANCES ------------------------------- #

inputs_A = keras.Input(shape=(image_height, image_width, 3))
inputs_B = keras.Input(shape=(image_height, image_width, 3))
model_A = keras.Model(inputs_A, outputs)
model_B = keras.Model(inputs_B, outputs)

# ------------------------------- CREATING SIAMESE MODEL ------------------------------- #
# https://medium.com/wicds/face-recognition-using-siamese-networks-84d6f2e54ea4

def euclidean_distance(vectors):
    (featA, featB) = vectors
    sum_squared = k.sum(k.square(featA - featB), axis=1, keepdims=True)
    return k.sqrt(k.maximum(sum_squared, k.epsilon()))

distance = keras.layers.Lambda(euclidean_distance)([model_A, model_B])
outputs = keras.layers.Dense(1, activation="sigmoid")(distance)
siamese_model = keras.Model(inputs=[inputs_A, inputs_B], outputs=outputs)
keras.utils.plot_model(siamese_model, to_file=f'{model_name}.png', show_layer_activations=True)


siamese_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# ------------------------------- TRAINING THE MODEL ------------------------------- #

history = siamese_model.fit(
            [train_image_pairs[:, 0], train_image_pairs[:, 1]],
            train_labels[:],
            validation_split=validation_split,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks
            )

# ------------------------------- SAVING THE MODEL ------------------------------- #

siamese_model.save(savedModelPath)