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
    
data_augmentation = keras.Sequential(
    [
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomBrightness(0.2),
    ]
)

def euclidean_distance(vectors):
    featA, featB = vectors
    sum_squared = k.sum(k.square(featA - featB), axis=1, keepdims=True)
    dstnc = k.sqrt(k.maximum(sum_squared, k.epsilon()))
    return dstnc

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss_with_margin(margin):
    def contrastive_loss(y_true, y_pred):
        square_pred = k.square(y_pred)
        margin_square = k.square(k.maximum(margin - y_pred, 0))
        return (y_true * square_pred + (1 - y_true) * margin_square)
    return contrastive_loss

def contrastive_loss_with_margin2(margin):
    def contrastive_loss(y_true, y_pred):
        square_pred = k.square(y_pred)
        margin_square = k.maximum(k.square(margin) - k.square(y_pred), 0)
        return (y_true * square_pred + (1 - y_true) * margin_square)
    return contrastive_loss

def contrastive_loss_with_margin_alt(margin):
    def contrastive_loss(y_true, y_pred):
        # explicitly cast the true class label data type to the predicted
        # class label data type (otherwise we run the risk of having two
        # separate data types, causing TensorFlow to error out)
        y_true = tf.cast(y_true, y_pred.dtype)
        # calculate the contrastive loss between the true labels and
        # the predicted labels
        squaredPreds = k.square(y_pred)
        squaredMargin = k.square(k.maximum(margin - y_pred, 0))
        loss = k.mean(y_true * squaredPreds + (1 - y_true) * squaredMargin)
        # return the computed contrastive loss to the calling function
        return loss
    return contrastive_loss
# ------------------------------- CNN MODELS ------------------------------- #

def MobileNet_WithTop_Weighted(input_shape, width_multiplier, depth_multiplier, dropoutRate, doDataAugmentation=False):
    # Loading either the MobileNet architecture model, and freeze it for transfer learning
    base = MobileNet(
                    input_shape=input_shape, # Optional shape tuple, only to be specified if include_top is False
                    alpha=width_multiplier, # Controls the width of the network. (Width multiplier)
                    depth_multiplier=depth_multiplier, # Depth multiplier for depthwise convolution. (Resolution multiplier)
                    dropout=dropoutRate, # Dropout rate. Default to 0.001.
                    weights="imagenet",
                    # weights=None,
                    include_top=False
                    )

    base.trainable = False

    inputs = keras.Input(shape=input_shape)
    # Data Augmentation on input
    if(doDataAugmentation):
        inputs = data_augmentation(inputs)

    base_model = base(inputs, training=False)
    outputs = MobileNet_Top(base_model, dropoutRate)

    return keras.Model(inputs, outputs)

def MobileNet_WithTop_NoWeights(input_shape, width_multiplier, depth_multiplier, dropoutRate, doDataAugmentation=False):
    # Loading either the MobileNet architecture model, and freeze it for transfer learning
    base = MobileNet(
                    input_shape=input_shape, # Optional shape tuple, only to be specified if include_top is False
                    alpha=width_multiplier, # Controls the width of the network. (Width multiplier)
                    depth_multiplier=depth_multiplier, # Depth multiplier for depthwise convolution. (Resolution multiplier)
                    dropout=dropoutRate, # Dropout rate. Default to 0.001.
                    weights=None,
                    # weights=None,
                    include_top=False
                    )

    inputs = keras.Input(shape=input_shape)
    # Data Augmentation on input
    if(doDataAugmentation):
        inputs = data_augmentation(inputs)

    base_model = base(inputs, training=False)
    outputs = MobileNet_Top(base_model, dropoutRate)

    return keras.Model(inputs, outputs)

def MobileNet_Top(base_model, dropout_rate):
    # Running base model in inference mode
    top_model = keras.layers.GlobalAveragePooling2D()(base_model)
    top_model = keras.layers.Dense(1024, kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=1e-4))(top_model)
    top_model = keras.layers.BatchNormalization()(top_model)
    top_model = keras.layers.Dropout(dropout_rate)(top_model)

    # Add Dense layer
    top_model = tf.keras.layers.Dense(256, kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=1e-4), activation='relu')(top_model)
    top_model = keras.layers.BatchNormalization()(top_model)
    #top_model = keras.layers.Activation('relu')(top_model)
    top_model = keras.layers.Dropout(dropout_rate)(top_model)

    outputs = tf.keras.layers.Dense(128, activation='relu')(top_model)

    return outputs

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
    x = keras.layers.Dropout(0.3)(x)
    # pooledOutput = keras.layers.Dense(1024)(pooledOutput)
    pooledOutput = keras.layers.Dense(512, kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=1e-4), activation='relu')(pooledOutput)
    outputs = keras.layers.Dense(128, kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=1e-4), activation='relu')(pooledOutput)

    model = keras.Model(inputs, outputs)
    return model

# ------------------------------- SIAMESE MODELS ------------------------------- #

# https://medium.com/wicds/face-recognition-using-siamese-networks-84d6f2e54ea4
def createSiameseModel(base_model, input_shape, doDataAugmentation=False):
    inputs_A = keras.Input(shape=input_shape)
    inputs_B = keras.Input(shape=input_shape)

    model_A = base_model(inputs_A)
    model_B = base_model(inputs_B)

    distance = keras.layers.Lambda(euclidean_distance)([model_A, model_B])
    outputs = keras.layers.Dense(1, activation='sigmoid')(distance)

    return keras.Model(inputs=[inputs_A, inputs_B], outputs=outputs)

# https://pub.towardsai.net/how-to-create-a-siamese-network-with-keras-to-compare-images-5713b3ee7a28
def createSiameseModel_alt(base_model, input_shape):
    inputs_A = keras.Input(shape=input_shape)
    inputs_B = keras.Input(shape=input_shape)
    
    modelA_output = base_model(inputs_A)
    modelB_output = base_model(inputs_B)

    outputs = keras.layers.Lambda(euclidean_distance, name='output_layer', 
                                    output_shape=eucl_dist_output_shape)([modelA_output, modelB_output])

    return keras.Model(inputs=[inputs_A, inputs_B], outputs=outputs)

def createSiameseModel_fromScratch(input_shape, doDataAugmentation=False):
    inputs = keras.Input(shape=input_shape)

    # Data Augmentation on input
    if(doDataAugmentation):
        inputs = data_augmentation(inputs)

    model = CNN(inputs)
    siamese_model = createSiameseModel(model, input_shape)

    return siamese_model

def createSiameseModel_fromScratch_alt(input_shape, doDataAugmentation=False):
    inputs = keras.Input(shape=input_shape)

    # Data Augmentation on input
    if(doDataAugmentation):
        inputs = data_augmentation(inputs)

    model = CNN(inputs)
    siamese_model = createSiameseModel_alt(model, input_shape)

    return siamese_model

    
def createSiameseModel_mobilenet_weighted(input_shape, width_multiplier, depth_multiplier, dropoutRate, doDataAugmentation=False):
    inputs = keras.Input(shape=input_shape)

    model = MobileNet_WithTop_Weighted(input_shape, width_multiplier, depth_multiplier, dropoutRate, doDataAugmentation)
    siamese_model = createSiameseModel(model, input_shape)

    return siamese_model
    
def createSiameseModel_mobilenet_noWeights(input_shape, width_multiplier, depth_multiplier, dropoutRate, doDataAugmentation=False):
    inputs = keras.Input(shape=input_shape)

    model = MobileNet_WithTop_NoWeights(input_shape, width_multiplier, depth_multiplier, dropoutRate, doDataAugmentation)
    siamese_model = createSiameseModel(model, input_shape)

    return siamese_model
        
def createSiameseModel_mobilenet_alt(input_shape, width_multiplier, depth_multiplier, dropoutRate, doDataAugmentation=False):
    inputs = keras.Input(shape=input_shape)

    # Data Augmentation on input
    if(doDataAugmentation):
        inputs = data_augmentation(inputs)

    model = MobileNet_WithTop(input_shape, width_multiplier, depth_multiplier, dropoutRate, doDataAugmentation)
    siamese_model = createSiameseModel(model, input_shape)

    return siamese_model