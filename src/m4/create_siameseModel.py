# ------------------------------- IMPORTS ------------------------------- #

# Tensorflow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNet, ResNet50

# ------------------------------- FUNCTIONS ------------------------------- #
    
data_augmentation = keras.Sequential(
    [
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomBrightness(0.2),
    ]
)

def euclidean_distance(vectors):
    """ Calculates the euclidian distance between two vectors

    Key arguments:
        vectors -- The two vectors to calculate the distance from
    
    Returns:
        euclidian_distance
    """

    x, y = vectors
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

def triplet_loss(margin=0.2):
    """ Calculates the Triplet Loss of the actual and predicted values.

    Key arguments:
        margin -- The margin value
    
    Returns:
        triplet_loss value
    """

    def loss(y_true, y_pred):
        anchor, positive, negative = y_pred[:,:128], y_pred[:,128:2*128], y_pred[:,2*128:]
        positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
        negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
        return tf.maximum(positive_dist - negative_dist + margin, 0.)
    return loss

def contrastive_loss_with_margin(margin):
    """ Calculates the Contrastive Loss of the actual and predicted values.

    Key arguments:
        margin -- The margin value
    
    Returns:
        contrastive_loss value
    """

    def contrastive_loss(y_true, y_pred):
        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean((1 - y_true) * square_pred + (y_true) * margin_square)
    return contrastive_loss

def freeze_layers(model, freeze_layers_percentage):
    """ Freezes the given percentage of layers of the model.

    Key arguments:
        model -- The model of which the layers are frozen
        freeze_layers_percentage -- The percentage of layers that should be frozen
    
    """
    for i in range(int(len(model.layers)*freeze_layers_percentage)):
        model.layers[i].trainable = False

# ------------------------------- CNN MODELS ------------------------------- #

def MobileNet_Top(base_model, dropout_rate, as_triplet=False):
    top_model = keras.layers.GlobalAveragePooling2D()(base_model)
    top_model = keras.layers.Dropout(dropout_rate)(top_model)

    top_model = keras.layers.Dense(256, activation='relu')(top_model)
    top_model = keras.layers.Dropout(dropout_rate)(top_model)

    if as_triplet:
        outputs = tf.keras.layers.Dense(128, activation='sigmoid')(top_model)
    else:
        outputs = tf.keras.layers.Dense(128, activation='relu')(top_model)

    return outputs

def ResNet_Top(base_model, dropout_rate, as_triplet=False):
    top_model = keras.layers.GlobalAveragePooling2D()(base_model)
    top_model = keras.layers.BatchNormalization()(top_model)
    top_model = keras.layers.Activation('relu')(top_model)
    top_model = keras.layers.Dropout(dropout_rate)(top_model)

    top_model = tf.keras.layers.Dense(256, activation='relu')(top_model)
    top_model = keras.layers.BatchNormalization()(top_model)
    top_model = keras.layers.Activation('relu')(top_model)
    top_model = keras.layers.Dropout(dropout_rate)(top_model)

    if as_triplet:
        outputs = tf.keras.layers.Dense(128, activation='sigmoid')(top_model)
    else:
        outputs = tf.keras.layers.Dense(128, activation='relu')(top_model)

    return outputs

def MobileNet_WithTop(input_shape, width_multiplier, depth_multiplier, dropoutRate, doDataAugmentation=False, use_weights=False, freeze_percentage=0.0, as_triplet=False):
    weights = 'imagenet' if use_weights else None

    # Loading either the MobileNet architecture model, and freeze it for transfer learning
    base = MobileNet(
                    input_shape=input_shape, # Optional shape tuple, only to be specified if include_top is False
                    alpha=width_multiplier, # Controls the width of the network. (Width multiplier)
                    depth_multiplier=depth_multiplier, # Depth multiplier for depthwise convolution. (Resolution multiplier)
                    dropout=dropoutRate, # Dropout rate. Default to 0.001.
                    weights=weights,
                    include_top=False
                    )

    freeze_layers(base, freeze_percentage)

    inputs = keras.Input(shape=input_shape)

    # Data Augmentation on input
    if(doDataAugmentation):
        inputs = data_augmentation(inputs)

    base_model = base(inputs)
    outputs = MobileNet_Top(base_model, dropoutRate, as_triplet)

    return keras.Model(inputs, outputs)

def ResNet_WithTop(input_shape, dropoutRate, doDataAugmentation=False, use_weights=False, freeze_percentage=0.0, as_triplet=False):
    weights = 'imagenet' if use_weights else None

    # Loading either the MobileNet architecture model, and freeze it for transfer learning
    base = ResNet50(
                    input_shape=input_shape, # Optional shape tuple, only to be specified if include_top is False
                    weights=weights,
                    include_top=False
                    )

    freeze_layers(base, freeze_percentage)

    inputs = keras.Input(shape=input_shape)

    # Data Augmentation on input
    if(doDataAugmentation):
        inputs = data_augmentation(inputs)

    base_model = base(inputs, training=False)
    outputs = ResNet_Top(base_model, dropoutRate, as_triplet)

    return keras.Model(inputs, outputs)

def CNN(input_shape):
    input = keras.layers.Input(input_shape)
    x = keras.layers.BatchNormalization()(input)
    x = keras.layers.Conv2D(64, (5, 5), activation='relu')(x)
    x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Conv2D(128, (5, 5), activation='relu')(x)
    x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(128, activation='relu')(x)

    return keras.Model(input, x)

# ------------------------------- SIAMESE MODELS ------------------------------- #

# https://medium.com/wicds/face-recognition-using-siamese-networks-84d6f2e54ea4
def SiameseModel(base_model, input_shape):
    inputs_A = keras.Input(shape=input_shape, name='input_1')
    inputs_B = keras.Input(shape=input_shape, name='input_2')

    model_A = base_model(inputs_A)
    model_B = base_model(inputs_B)

    distance = keras.layers.Lambda(euclidean_distance)([model_A, model_B])
    distance = keras.layers.BatchNormalization()(distance)
    outputs = keras.layers.Dense(1, activation='sigmoid')(distance)

    return keras.Model(inputs=[inputs_A, inputs_B], outputs=outputs)


def SiameseModel_Triplet(base_model, input_shape):
    inputs_anchor = keras.Input(shape=input_shape)
    inputs_positives = keras.Input(shape=input_shape)
    inputs_negatives = keras.Input(shape=input_shape)
    
    anchor_model_output = base_model(inputs_anchor)
    positives_model_output = base_model(inputs_positives)
    negatives_model_output = base_model(inputs_negatives)

    outputs = keras.layers.concatenate([anchor_model_output, positives_model_output, negatives_model_output])

    return keras.Model(inputs=[inputs_anchor, inputs_positives, inputs_negatives], outputs=outputs)

# ------------------------------- USER FUNCTIONS FOR MODELS CREATION ------------------------------- #

def createSiameseModel_fromScratch(input_shape, dropoutRate, doDataAugmentation=False, as_triplet=False):
    """ Creates a siamese model with a from scratch model as the base model.

    Key arguments:
        input_shape -- The input shape for the model
        dropoutRate -- Dropout rate
        doDataAugmentation -- Indication whether data augmentation should be performed (default: False)
        as_triplet -- Indication whether the model should be created as a Triplet Loss model (default: False)

    Returns:
        siamese_model
    """

    model = CNN(input_shape)
    
    if as_triplet:
        siamese_model = SiameseModel_Triplet(model, input_shape)
    else:
        siamese_model = SiameseModel(model, input_shape)

    return siamese_model

def createSiameseModel_mobilenet(input_shape, width_multiplier, depth_multiplier, dropoutRate, doDataAugmentation=False, use_weights=False, freeze_percentage=1.0, as_triplet=False):
    """ Creates a siamese model with the MobileNet model as the base model.

    Key arguments:
        input_shape -- The input shape for the model
        width_multiplier -- MobileNet width multiplier
        depth_multiplier -- MobileNet depth multiplier
        dropoutRate -- Dropout rate
        doDataAugmentation -- Indication whether data augmentation should be performed (default: False)
        use_weights=False -- Indication whether ImageNet weights should be loaded into the base model
        freeze_percentage -- The percentage of frozen layer in the base model (default: 1.0)
        as_triplet -- Indication whether the model should be created as a Triplet Loss model (default: False)

    Returns:
        siamese_model
    """

    model = MobileNet_WithTop(
                input_shape,
                width_multiplier,
                depth_multiplier,
                dropoutRate,
                doDataAugmentation,
                use_weights=use_weights,
                freeze_percentage=freeze_percentage,
                as_triplet=as_triplet
                )

    if as_triplet:
        siamese_model = SiameseModel_Triplet(model, input_shape)
    else:
        siamese_model = SiameseModel(model, input_shape)

    return siamese_model
  
def createSiameseModel_resnet(input_shape, dropoutRate, doDataAugmentation=False, use_weights=False, freeze_percentage=1.0, as_triplet=False):
    """ Creates a siamese model with the ResNet50 model as the base model.

    Key arguments:
        input_shape -- The input shape for the model
        dropoutRate -- Dropout rate
        doDataAugmentation -- Indication whether data augmentation should be performed (default: False)
        use_weights=False -- Indication whether ImageNet weights should be loaded into the base model
        freeze_percentage -- The percentage of frozen layer in the base model (default: 1.0)
        as_triplet -- Indication whether the model should be created as a Triplet Loss model (default: False)

    Returns:
        siamese_model
    """

    model = ResNet_WithTop(
                input_shape, 
                dropoutRate,
                doDataAugmentation,
                use_weights=use_weights,
                freeze_percentage=freeze_percentage,
                as_triplet=as_triplet,
                )

    if as_triplet:
        siamese_model = SiameseModel_Triplet(model, input_shape)
    else:
        siamese_model = SiameseModel(model, input_shape)

    return siamese_model