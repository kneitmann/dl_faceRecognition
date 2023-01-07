# ------------------------------- IMPORTS ------------------------------- #

import os

from tensorflow import keras

from loadData import createDataset, generate_image_pairs
from create_siameseModel import createSiameseModel_mobilenet

# ------------------------------- PARAMETERS ------------------------------- #

# Log parameters
model_name = 'siamese_model_mobilenet'
savedModelPath = f'./log/saved_models/{model_name}'
tb_log_dir = f'./log/tensorboard/{model_name}'
cp_filepath = f'./log/cps/{model_name}/'
training_data_path = './data/m4/training/'

if not os.path.exists(cp_filepath):
    os.makedirs(cp_filepath)

# Dynamic hyperparameters
learningRate = 0.001
doDataAugmentation = False
dropoutRate = 0.25
width_multiplier = 1
depth_multiplier = 1

# Training parameters
batch_size = 4
epochs = 50
validation_split = 0.2

callbacks = [
    # Checkpoint callback                    
    keras.callbacks.ModelCheckpoint(
                    filepath=cp_filepath + 'latest_weights.h5', 
                    verbose=1, 
                    save_weights_only=True),

    # Tensorboard callback
    keras.callbacks.TensorBoard(log_dir=tb_log_dir, histogram_freq=1)
]

# Data parameters
image_height = 160
image_width = 160

# ----------------------------------- DATASETS ----------------------------------- #

# Creating the training and validation dataset
train_images, train_labels = createDataset(training_data_path, (image_height, image_width))
train_image_pairs, train_pair_labels = generate_image_pairs(train_images, train_labels)

# ------------------------------- CREATING AND COMPILING THE MODEL ------------------------------- #

siamese_model = createSiameseModel_mobilenet((image_height, image_width, 3), width_multiplier, depth_multiplier, dropoutRate, doDataAugmentation)

keras.utils.plot_model(siamese_model, to_file=f'{model_name}.png', show_layer_activations=True)
siamese_model.summary()

siamese_model.compile(loss='binary_crossentropy', optimizer='adam', metrics='binary_accuracy')


# ------------------------------- TRAINING THE MODEL ------------------------------- #

history = siamese_model.fit(
            [train_image_pairs[:, 0], train_image_pairs[:, 1]],
            train_pair_labels[:],
            validation_split=validation_split,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks
            )


# ------------------------------- SAVING THE MODEL ------------------------------- #

siamese_model.save_weights(cp_filepath + f'{model_name}.h5')