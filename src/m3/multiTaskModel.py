import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNet
import numpy as np


def createfaceDetectTopModel_mobilenet(base_model, model_name, input_shape, width_multiplier=1, depth_multiplier=1, dropoutRate=0.0):
    inputs = keras.Input(shape=input_shape)

    # Data Augmentation on input
    # if(doDataAugmentation):
    #     inputs = data_augmentation(inputs)

    # Running base model in inference mode
    model = base_model(inputs, training=False)

    model = keras.layers.Flatten()(model)
    model = keras.layers.Dense(256, activation='sigmoid')(model)
    model = keras.layers.Dense(1, activation='sigmoid', name='face_output')(model)

    return model


def createMaskHead(base_model):
    # Mask Head
    mask_head = tf.keras.layers.Dense(512)(base_model)
    mask_head = tf.keras.layers.Dense(256)(mask_head)

    ## Final layer for binary classification
    mask_outputs = keras.layers.Dense(1, activation='sigmoid', name='mask_output')(mask_head)
    return mask_outputs

def createAgeHead(base_model):
    # Age Head
    age_head = tf.keras.layers.Dense(1024)(base_model)
    age_head = tf.keras.layers.Dense(512)(age_head)
    age_head = tf.keras.layers.Dense(256)(age_head)

    ## Final layer for classification
    age_outputs = keras.layers.Dense(122, activation='softmax', name='age_output')(age_head)

    return age_outputs
