# ------------------------------- IMPORTS ------------------------------- #

import os
import matplotlib.pyplot as plt

# Tensorflow imports
import tensorflow as tf
from tensorflow import keras
import keras.backend as k
#from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import MobileNet

from loadData import createDataset, load_img

# ------------------------------- FUNCTIONS ------------------------------- #

def euclidean_distance(vectors):
    (featA, featB) = vectors
    sum_squared = k.sum(k.square(featA - featB), axis=1, keepdims=True)
    dstnc = k.sqrt(k.maximum(sum_squared, k.epsilon()))
    return dstnc

    
data_augmentation = keras.Sequential(
    [
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomBrightness(0.2),
    ]
)

# ------------------------------- CNN MODELS ------------------------------- #

def MobileNet_WithTop(input_shape, width_multiplier, depth_multiplier, dropoutRate, doDataAugmentation=False):
    # Loading either the MobileNet architecture model, and freeze it for transfer learning
    base = MobileNet(
                    input_shape=input_shape, # Optional shape tuple, only to be specified if include_top is False
                    alpha=width_multiplier, # Controls the width of the network. (Width multiplier)
                    depth_multiplier=depth_multiplier, # Depth multiplier for depthwise convolution. (Resolution multiplier)
                    dropout=dropoutRate, # Dropout rate. Default to 0.001.
                    weights="imagenet",
                    include_top=False
                    )

    base.trainable = False

    inputs = keras.Input(shape=input_shape)

    # Data Augmentation on input
    if(doDataAugmentation):
        inputs = data_augmentation(inputs)

    # Running base model in inference mode
    base_model = base(inputs, training=False)
    top_model = keras.layers.GlobalAveragePooling2D()(base_model)
    top_model = keras.layers.Dense(1024)(top_model)
    #base_model = keras.layers.Dropout(0.3)(base_model)

    # Add Dense layer
    top_model = tf.keras.layers.Dense(256, activation='relu')(top_model)
    #top_model = keras.layers.BatchNormalization()(top_model)
    #top_model = keras.layers.Activation('relu')(top_model)
    #top_model = keras.layers.Dropout(0.2)(top_model)

    outputs = tf.keras.layers.Dense(128, activation='relu')(top_model)

    return keras.Model(inputs, outputs)

# https://keras.io/examples/vision/image_classification_from_scratch/#build-a-model

def Xception_Small(inputs):
    # Entry block
    x = keras.layers.Conv2D(128, 3, strides=2, padding="same")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    for size in [256, 512, 728]:
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.SeparableConv2D(size, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Activation("relu")(x)
        x = keras.layers.SeparableConv2D(size, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

    x = keras.layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(256)(x)
    outputs = keras.layers.Dense(128)(x)

    return keras.Model(inputs, outputs)

def CNN(inputs):
    # x = keras.layers.Conv2D(128, (11, 11), padding="same", activation="relu")(inputs)
    # x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    # x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Conv2D(256, (5, 5), padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Dropout(0.3)(x)

    # x = keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu")(x)
    # x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    # x = keras.layers.Dropout(0.3)(x)

    pooledOutput = keras.layers.GlobalAveragePooling2D()(x)
    # pooledOutput = keras.layers.Dense(1024)(pooledOutput)
    pooledOutput = keras.layers.Dense(512)(pooledOutput)
    outputs = keras.layers.Dense(128)(pooledOutput)

    model = keras.Model(inputs, outputs)
    return model

# ------------------------------- SIAMESE MODELS ------------------------------- #

# https://medium.com/wicds/face-recognition-using-siamese-networks-84d6f2e54ea4
def createSiameseModel(base_model, input_shape):
    inputs_A = keras.Input(shape=input_shape)
    inputs_B = keras.Input(shape=input_shape)
    model_A = base_model(inputs_A)
    model_B = base_model(inputs_B)

    distance = keras.layers.Lambda(euclidean_distance)([model_A, model_B])
    outputs = keras.layers.Dense(1, activation='sigmoid')(distance)

    return keras.Model(inputs=[inputs_A, inputs_B], outputs=outputs)

def createSiameseModel_fromScratch(input_shape, doDataAugmentation=False):
    inputs = keras.Input(shape=input_shape)

    # Data Augmentation on input
    if(doDataAugmentation):
        inputs = data_augmentation(inputs)

    model = CNN(inputs)
    siamese_model = createSiameseModel(model, input_shape)

    return siamese_model

    
def createSiameseModel_mobilenet(input_shape, width_multiplier, depth_multiplier, dropoutRate, doDataAugmentation=False):
    inputs = keras.Input(shape=input_shape)

    # Data Augmentation on input
    if(doDataAugmentation):
        inputs = data_augmentation(inputs)

    model = MobileNet_WithTop(input_shape, width_multiplier, depth_multiplier, dropoutRate, doDataAugmentation)
    siamese_model = createSiameseModel(model, input_shape)

    return siamese_model