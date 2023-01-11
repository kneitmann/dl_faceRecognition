# ------------------------------- IMPORTS ------------------------------- #

import os
import numpy as np

from tensorflow import keras
import keras.backend as k
import tensorflow as tf

from loadData import createDataset, generate_image_pairs
from create_siameseModel import createSiameseModel_mobilenet_weighted, createSiameseModel_mobilenet_noWeights, contrastive_loss_with_margin, contrastive_loss_with_margin2

# ------------------------------- PARAMETERS ------------------------------- #

# Log parameters
model_name = 'siamese_model_mobilenet_binaryloss'
savedModelPath = f'./log/saved_models/{model_name}/'
tb_log_dir = f'./log/tensorboard/{model_name}/'
cp_filepath = f'./log/cps/{model_name}/'
training_data_path = './data/m4_manyOne/training/'
validation_data_path = './data/m4_manyOne/validation/'

if not os.path.exists(cp_filepath):
    os.makedirs(cp_filepath)

if not os.path.exists(savedModelPath):
    os.makedirs(savedModelPath)

# Dynamic hyperparameters
learningRate = 0.001
doDataAugmentation = False
dropoutRate = 0.3
width_multiplier = 1
depth_multiplier = 1

# Training parameters
batch_size = 64
epochs = 100
validation_split = 0.2
useWeights = False
decay = learningRate/epochs

def lr_time_decay(epoch, lr):
    return lr*1/(1+decay*epoch)

callbacks = [
    # Checkpoint callback                    
    keras.callbacks.ModelCheckpoint(
                    filepath=cp_filepath + 'latest_weights.h5', 
                    verbose=1, 
                    save_weights_only=True),

    # Tensorboard callback
    keras.callbacks.TensorBoard(log_dir=tb_log_dir, histogram_freq=1),

    # Learning Rate decay
    # keras.callbacks.LearningRateScheduler(lr_time_decay, verbose=1),
    
    keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.001,
            patience=5,
            verbose=1,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
            start_from_epoch=20,
            )
]

# Data parameters
image_height = 160
image_width = 160

# ----------------------------------- DATASETS ----------------------------------- #

# Creating the training and validation dataset
train_images, train_labels = createDataset(training_data_path, (image_height, image_width))
val_images, val_labels = createDataset(validation_data_path, (image_height, image_width))

# Creating the training image pairs and the corrsponding labels
train_image_pairs, train_pair_labels = generate_image_pairs(train_images, train_labels)
val_image_pairs, val_pair_labels = generate_image_pairs(val_images, val_labels)

# Shuffling the training data
np.random.seed(42)
np.random.shuffle(train_image_pairs)
np.random.seed(42)
np.random.shuffle(train_pair_labels)

np.random.seed(42)
np.random.shuffle(val_image_pairs)
np.random.seed(42)
np.random.shuffle(val_pair_labels)

# ------------------------------- CREATING AND COMPILING THE MODEL ------------------------------- #

if useWeights:
    siamese_model = createSiameseModel_mobilenet_weighted((image_height, image_width, 3), width_multiplier, depth_multiplier, dropoutRate, doDataAugmentation)
else:
    siamese_model = createSiameseModel_mobilenet_noWeights((image_height, image_width, 3), width_multiplier, depth_multiplier, dropoutRate, doDataAugmentation)

keras.utils.plot_model(siamese_model, to_file=f'siamese_model.png', show_layer_activations=True)
siamese_model.summary()

siamese_model.compile(
            # loss=contrastive_loss_with_margin(margin=0.3),
            loss='binary_crossentropy',
            metrics='accuracy',
            optimizer=keras.optimizers.Adam(learning_rate=learningRate), 
            )


# ------------------------------- TRAINING THE MODEL ------------------------------- #

history = siamese_model.fit(
            [train_image_pairs[:, 0], train_image_pairs[:, 1]],
            train_pair_labels[:],
            validation_data=([val_image_pairs[:, 0], val_image_pairs[:, 1]], val_pair_labels),
            # validation_split=validation_split,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks
            )


# ------------------------------- SAVING THE MODEL ------------------------------- #

siamese_model.save_weights(savedModelPath + f'{model_name}.h5')