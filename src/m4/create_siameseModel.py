# ------------------------------- IMPORTS ------------------------------- #

import os
import matplotlib.pyplot as plt

# Tensorflow imports
import tensorflow as tf
from tensorflow import keras
import keras.backend as k
#from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import MobileNet, ResNet50

from loadData import createDataset, load_img

# ------------------------------- FUNCTIONS ------------------------------- #
    
data_augmentation = keras.Sequential(
    [
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomBrightness(0.2),
    ]
)

def euclidean_distance(vectors):
    x, y = vectors
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss_with_margin(margin):
    def contrastive_loss(y_true, y_pred):
        square_pred = k.square(y_pred)
        margin_square = k.square(k.maximum(margin - y_pred, 0))
        return (y_true * square_pred + (1 - y_true) * margin_square)
    return contrastive_loss

def contrastive_loss_with_margin_alt(margin):
    def contrastive_loss(y_true, y_pred):
        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean((1 - y_true) * square_pred + (y_true) * margin_square)
    return contrastive_loss
# ------------------------------- CNN MODELS ------------------------------- #

def MobileNet_Top(base_model, dropout_rate):
    # top_model = keras.layers.BatchNormalization()(base_model)
    # top_model = keras.layers.Activation('relu')(top_model)
    top_model = keras.layers.GlobalAveragePooling2D()(base_model)
    top_model = keras.layers.Dense(1024)(top_model)
    # top_model = keras.layers.BatchNormalization()(top_model)
    # top_model = keras.layers.Activation('relu')(top_model)
    # top_model = keras.layers.Dropout(dropout_rate)(top_model)

    # Add Dense layer
    top_model = tf.keras.layers.Dense(256, activation='relu')(top_model)
    # top_model = keras.layers.BatchNormalization()(top_model)
    # top_model = keras.layers.Activation('relu')(top_model)
    # top_model = keras.layers.Dropout(dropout_rate)(top_model)

    outputs = tf.keras.layers.Dense(128, activation='relu')(top_model)

    return outputs

def ResNet_Top(base_model, dropout_rate):
    # top_model = keras.layers.BatchNormalization()(base_model)
    # top_model = keras.layers.Activation('relu')(top_model)
    top_model = keras.layers.GlobalAveragePooling2D()(base_model)
    # top_model = keras.layers.Dense(1024, kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=1e-4))(top_model)
    top_model = keras.layers.BatchNormalization()(top_model)
    top_model = keras.layers.Activation('relu')(top_model)
    top_model = keras.layers.Dropout(dropout_rate)(top_model)

    top_model = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=1e-4))(top_model)
    top_model = keras.layers.BatchNormalization()(top_model)
    top_model = keras.layers.Activation('relu')(top_model)
    top_model = keras.layers.Dropout(dropout_rate)(top_model)

    outputs = tf.keras.layers.Dense(128, activation='relu')(top_model)

    return outputs

def MobileNet_WithTop_Weighted(input_shape, width_multiplier, depth_multiplier, dropoutRate, doDataAugmentation=False):
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

    base_model = base(inputs)
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
                    include_top=False
                    )

    inputs = keras.Input(shape=input_shape)

    # Data Augmentation on input
    if(doDataAugmentation):
        inputs = data_augmentation(inputs)

    base_model = base(inputs)
    outputs = MobileNet_Top(base_model, dropoutRate)

    return keras.Model(inputs, outputs)

def ResNet_WithTop_NoWeights(input_shape, dropoutRate, doDataAugmentation=False):
    # Loading either the ResNet architecture model
    base = ResNet50(
                    input_shape=input_shape, # Optional shape tuple, only to be specified if include_top is False
                    weights=None,
                    include_top=False
                    )

    inputs = keras.Input(shape=input_shape)

    # Data Augmentation on input
    if(doDataAugmentation):
        inputs = data_augmentation(inputs)

    base_model = base(inputs)
    outputs = ResNet_Top(base_model, dropoutRate)

    return keras.Model(inputs, outputs)


def ResNet_WithTop_Weighted(input_shape, dropoutRate, doDataAugmentation=False):
    # Loading either the MobileNet architecture model, and freeze it for transfer learning
    base = ResNet50(
                    input_shape=input_shape, # Optional shape tuple, only to be specified if include_top is False
                    weights="imagenet",
                    include_top=False
                    )

    base.trainable = False

    inputs = keras.Input(shape=input_shape)

    # Data Augmentation on input
    if(doDataAugmentation):
        inputs = data_augmentation(inputs)

    base_model = base(inputs, training=False)
    outputs = ResNet_Top(base_model, dropoutRate)

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

def CNN(input_shape):
    inputs = keras.Input(shape=input_shape)
    inputs = keras.layers.Rescaling(1./255)(inputs)
    x = keras.layers.BatchNormalization()(inputs)
    # x = keras.layers.Conv2D(128, (11, 11), padding="same", activation="relu")(inputs)
    # x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    # x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Conv2D(256, (5, 5), padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Dropout(0.3)(x)

    # x = keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu")(x)
    # x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    # x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    # pooledOutput = keras.layers.Dense(1024)(pooledOutput)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    outputs = keras.layers.Dense(128, activation='relu')(x)

    model = keras.Model(inputs, outputs)
    return model

def CNN2(input_shape):
    input = keras.layers.Input(input_shape)
    input = keras.layers.Rescaling(1./255)(input)
    x = keras.layers.BatchNormalization()(input)
    x = keras.layers.Conv2D(64, (5, 5), activation="tanh")(x)
    x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Conv2D(128, (5, 5), activation="tanh")(x)
    x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Flatten()(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(128, activation="tanh")(x)

    return keras.Model(input, x)

# ------------------------------- SIAMESE MODELS ------------------------------- #

# https://medium.com/wicds/face-recognition-using-siamese-networks-84d6f2e54ea4
def createSiameseModel(base_model, input_shape, doDataAugmentation=False):
    inputs_A = keras.Input(shape=input_shape, name='input_1')
    inputs_B = keras.Input(shape=input_shape, name='input_2')

    model_A = base_model(inputs_A)
    model_B = base_model(inputs_B)

    distance = keras.layers.Lambda(euclidean_distance)([model_A, model_B])
    distance = keras.layers.BatchNormalization()(distance)
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
    # Data Augmentation on input
    if(doDataAugmentation):
        inputs = data_augmentation(inputs)

    model = CNN2(input_shape)
    siamese_model = createSiameseModel(model, input_shape)

    return siamese_model

def createSiameseModel_fromScratch_alt(input_shape, doDataAugmentation=False):
    model = CNN(input_shape)
    siamese_model = createSiameseModel_alt(model, input_shape)

    return siamese_model
  
def createSiameseModel_mobilenet(input_shape, width_multiplier, depth_multiplier, dropoutRate, doDataAugmentation=False, use_weights=False):
    if(use_weights):
        model = MobileNet_WithTop_Weighted(input_shape, width_multiplier, depth_multiplier, dropoutRate, doDataAugmentation)
    else:
        model = MobileNet_WithTop_NoWeights(input_shape, width_multiplier, depth_multiplier, dropoutRate, doDataAugmentation)

    siamese_model = createSiameseModel(model, input_shape)

    return siamese_model
        
def createSiameseModel_mobilenet_alt(input_shape, width_multiplier, depth_multiplier, dropoutRate, doDataAugmentation=False, use_weights=False):
    if(use_weights):
        model = MobileNet_WithTop_Weighted(input_shape, width_multiplier, depth_multiplier, dropoutRate, doDataAugmentation)
    else:
        model = MobileNet_WithTop_NoWeights(input_shape, width_multiplier, depth_multiplier, dropoutRate, doDataAugmentation)
    
    siamese_model = createSiameseModel_alt(model, input_shape)

    return siamese_model
  
def createSiameseModel_resnet(input_shape, dropoutRate, doDataAugmentation=False, use_weights=False):
    if(use_weights):
        model = ResNet_WithTop_Weighted(input_shape, dropoutRate, doDataAugmentation)
    else:
        model = ResNet_WithTop_NoWeights(input_shape, dropoutRate, doDataAugmentation)

    siamese_model = createSiameseModel(model, input_shape)

    return siamese_model
        
def createSiameseModel_resnet_alt(input_shape, dropoutRate, doDataAugmentation=False, use_weights=False):
    if(use_weights):
        model = ResNet_WithTop_Weighted(input_shape, dropoutRate, doDataAugmentation)
    else:
        model = ResNet_WithTop_NoWeights(input_shape, dropoutRate, doDataAugmentation)
    
    siamese_model = createSiameseModel_alt(model, input_shape)

    return siamese_model