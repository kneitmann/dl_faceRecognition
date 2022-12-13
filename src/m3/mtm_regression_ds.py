import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# Tensorflow imports
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications import ResNet50

from loadData_alt import createDataset

# Log parameters
model_name = 'mtm_regression_ds'
savedModelPath = f'../../log/saved_models/{model_name}'
tb_log_dir = f'../../log/tensorboard/{model_name}'
cp_filepath = f'../../log/cps/{model_name}/'

if not os.path.exists(cp_filepath):
    os.makedirs(cp_filepath)

cp_filepath += 'latest_weights.h5'

# Dynamic hyperparameters
learningRate = 0.001
doDataAugmentation = True
doFineTuning = False
dropoutRate = 0.25
width_multiplier = 1
depth_multiplier = 1

# Training parameters
batch_size = 32
epochs = 5

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

training_ds, validation_ds = createDataset('../../data/m3/training', (image_height, image_width), batch_size, 0.2, for_regression=True)

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

# Add Dense layer
face_head = tf.keras.layers.Dense(256, activation='relu')(base_model)
face_head = keras.layers.BatchNormalization()(face_head)
face_head = keras.layers.Activation('relu')(face_head)
face_head = keras.layers.Dropout(0.2)(face_head)
face_head = tf.keras.layers.Dense(128, activation='relu')(face_head)
face_head = tf.keras.layers.Dense(64, activation='relu')(face_head)

# Final layer for binary classification
face_outputs = keras.layers.Dense(1, activation='sigmoid', name='face_output')(face_head)

# Add Dense layer
mask_head = tf.keras.layers.Dense(256, activation='relu')(base_model)
mask_head = keras.layers.BatchNormalization()(mask_head)
mask_head = keras.layers.Activation('relu')(mask_head)
mask_head = keras.layers.Dropout(0.2)(mask_head)
mask_head = tf.keras.layers.Dense(128, activation='relu')(mask_head)
mask_head = tf.keras.layers.Dense(64, activation='relu')(mask_head)

# Final layer for binary classification
mask_outputs = keras.layers.Dense(1, activation='sigmoid', name='mask_output')(mask_head)

# Add Dense layer
age_head = tf.keras.layers.Dense(256, activation='relu')(base_model)
age_head = keras.layers.BatchNormalization()(age_head)
age_head = keras.layers.Activation('relu')(age_head)
age_head = keras.layers.Dropout(0.2)(age_head)
age_head = tf.keras.layers.Dense(128, activation='relu')(age_head)
age_head = tf.keras.layers.Dense(64, activation='relu')(age_head)

# Final layer for binary classification
age_outputs = keras.layers.Dense(1, activation='relu', name='age_output')(age_head)


model = keras.Model(inputs, [face_outputs, mask_outputs, age_outputs])
#keras.utils.plot_model(model)

# Using a joint loss function for the three tasks:
# [ Loss = gamma * Loss_task1 + gamma * Loss_task2 + gamma * Loss_task3 ]
# Because every task is dependant on every other task, the model receives the loss of every task when gamma > 0

gamma = 1

model.compile(
      optimizer=keras.optimizers.Adam(), # Learning Rate?
            loss={
                  'face_output': 'mean_squared_error', 
                  'mask_output': 'mean_squared_error',
                  'age_output': 'mean_squared_error'
                  },
            loss_weights={
                  'face_output': 0.33 * gamma,
                  'mask_output': 0.33 * gamma,
                  'age_output': 0.33 * gamma
                  }, 
            metrics={
                  'face_output': 'mean_absolute_error', 
                  'mask_output': 'mean_absolute_error',
                  'age_output': 'mean_absolute_error'
                  },
)

model.summary()

history = model.fit(
            training_ds,
            validation_data=validation_ds,
            epochs=epochs, 
            callbacks=callbacks,
            shuffle=True
        )

model.save(savedModelPath)

# https://www.tensorflow.org/tutorials/images/classification

img = tf.keras.utils.load_img(
    '../../data/m3/training/face/noMask/1_0_19_501540.jpg', target_size=(image_height, image_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array_batch = tf.expand_dims(img_array, 0) # Create a batch

preds = model.predict(img_array_batch)

face_pred_percent = preds[0][0][0]*100
mask_pred_percent = preds[1][0][0]*100
age_pred = preds[2][0][0]

ax = plt.subplot(1, 1, 1)
plt.imshow(img)
plt.title("Face: {:.2f}% | Mask: {:.2f}% | Age: {:.0f}".format(face_pred_percent, mask_pred_percent, age_pred))