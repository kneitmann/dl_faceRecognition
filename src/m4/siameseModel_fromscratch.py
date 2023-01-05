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
batch_size = 4
epochs = 12
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

# ----------------------------------- FUNCTIONS ----------------------------------- #

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
    x = keras.layers.Conv2D(96, (11, 11), padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Conv2D(256, (5, 5), padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Conv2D(384, (3, 3), padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Dropout(0.3)(x)

    pooledOutput = keras.layers.GlobalAveragePooling2D()(x)
    pooledOutput = keras.layers.Dense(1024)(pooledOutput)
    outputs = keras.layers.Dense(128)(pooledOutput)

    model = keras.Model(inputs, outputs)
    return model

# ----------------------------------- DATASETS ----------------------------------- #

# Creating the training and validation dataset
train_image_pairs, train_labels = createDataset(training_data_path, (image_height, image_width), grayscale=True)


# ------------------------------- CREATING MODEL ------------------------------- #

inputs = keras.Input(shape=(image_height, image_width, 1))

# Data Augmentation on input
if(doDataAugmentation):
    inputs = data_augmentation(inputs)

model = CNN(inputs)

# ------------------------------- CREATING MODEL INSTANCES ------------------------------- #

inputs_A = keras.Input(shape=(image_height, image_width, 1))
inputs_B = keras.Input(shape=(image_height, image_width, 1))
model_A = model(inputs_A)
model_B = model(inputs_B)


# ------------------------------- CREATING SIAMESE MODEL ------------------------------- #
# https://medium.com/wicds/face-recognition-using-siamese-networks-84d6f2e54ea4

def euclidean_distance(vectors):
    (featA, featB) = vectors
    sum_squared = k.sum(k.square(featA - featB), axis=1, keepdims=True)
    dstnc = k.sqrt(k.maximum(sum_squared, k.epsilon()))
    return dstnc

distance = keras.layers.Lambda(euclidean_distance)([model_A, model_B])
outputs = keras.layers.Dense(1, activation='sigmoid')(distance)

siamese_model = keras.Model(inputs=[inputs_A, inputs_B], outputs=outputs)
keras.utils.plot_model(siamese_model, to_file=f'{model_name}.png', show_layer_activations=True)
siamese_model.summary()

siamese_model.compile(loss='binary_crossentropy', optimizer='adam', metrics='binary_accuracy')


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

keras.models.save_model(
    siamese_model,
    savedModelPath,
    # overwrite=True,
    # include_optimizer=True,
    # save_format=None,
    # signatures=None,
    # options=None,
    # save_traces=True
)

def get_img_predictions(model, img_paths, grayscale=False):
    # Loading and preprocessing the image
    color_mode = 'rgb'
    if grayscale: color_mode = 'grayscale'

    img1 = keras.utils.load_img(
                img_paths[0],
                color_mode=color_mode,
                target_size=(image_height, image_width),
                interpolation="bilinear",
                keep_aspect_ratio=True,
            )
    img2 = keras.utils.load_img(
                img_paths[1],
                color_mode=color_mode,
                target_size=(image_height, image_width),
                interpolation="bilinear",
                keep_aspect_ratio=True,
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

img1, img2, pred = get_img_predictions(siamese_model, img_paths, grayscale=True)
print(f'Similarity: {pred}')

# Showing the image with the corresponding predictions
ax = plt.subplot(2, 1, 1)
plt.imshow(img1)
plt.title("Same Face: {:.2f}%)".format(pred * 100, actual))