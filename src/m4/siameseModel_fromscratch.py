# ------------------------------- IMPORTS ------------------------------- #

import os
import matplotlib.pyplot as plt

# Tensorflow imports
import tensorflow as tf
from tensorflow import keras

from loadData import createDataset
from create_siameseModel import createSiameseModel_fromScratch

# ------------------------------- PARAMETERS ------------------------------- #

# Log parameters
model_name = 'siamese_model_fromScratch'
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

# ----------------------------------- DATASETS ----------------------------------- #

# Creating the training and validation dataset
train_image_pairs, train_labels = createDataset(training_data_path, (image_height, image_width), grayscale=True)


# ------------------------------- CREATING AND COMPILING MODEL ------------------------------- #

siamese_model = createSiameseModel_fromScratch((image_height, image_width, 1), doDataAugmentation)
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

# ------------------------------- MAKING A PREDICTION ------------------------------- #

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