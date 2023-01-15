# ------------------------------- IMPORTS ------------------------------- #

import os
import numpy as np

from tensorflow import keras

from loadData import createDataset, generate_image_pairs, generate_image_triplets
from create_siameseModel import MobileNet_WithTop, SiameseModel_Triplet, createSiameseModel_mobilenet, triplet_loss, contrastive_loss_with_margin, keras_triplet_loss

# ------------------------------- PARAMETERS ------------------------------- #

# Log parameters
model_name = 'siamese_model_mobilenet_weights_margin1,0_frozen0,75_triplet'
savedModelPath = f'./log/saved_models/{model_name}/'
tb_log_dir = f'./log/tensorboard/{model_name}/'
cp_filepath = f'./log/cps/{model_name}/'
training_data_path = './data/m4_manyOne/training/cropped/'
validation_data_path = './data/m4_manyOne/validation/'

if not os.path.exists(cp_filepath):
    os.makedirs(cp_filepath)

if not os.path.exists(savedModelPath):
    os.makedirs(savedModelPath)

# Hyperparameters

## Data parameters
image_height = 128
image_width = 128
validation_split = 0.2

## Training parameters
batch_size = 32
epochs = 30
learningRate = 0.001
loss = 'triplet_loss'
optimizer = keras.optimizers.Adam(learningRate) if loss == 'triplet_loss' else keras.optimizers.RMSprop(learningRate)
margin = 1.0
callbacks = [
    # Checkpoint callback                    
    keras.callbacks.ModelCheckpoint(
                    filepath=cp_filepath + 'latest_weights.h5', 
                    verbose=1, 
                    save_weights_only=True),

    # Tensorboard callback
    keras.callbacks.TensorBoard(log_dir=tb_log_dir, histogram_freq=1),

    keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.001,
            patience=5,
            verbose=1,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
            start_from_epoch=10,
            )
]

## Model parameters
data_augmentation = False
dropout_rate = 0.3
frozen_layers_percent = 0.75
use_weights = True
as_triplet = loss == 'triplet_loss'
width_multiplier = 1 # MobileNet specific
depth_multiplier = 1 # MobileNet specific

losses_dict = {
    'binary_crossentropy' : 'binary_crossentropy',
    'contrastive_loss' : contrastive_loss_with_margin(margin=margin),
    'triplet_loss' : triplet_loss(margin=margin)
}

# ----------------------------------- DATASETS ----------------------------------- #

# Creating the training and validation dataset
train_images, train_labels = createDataset(training_data_path, (image_height, image_width), preprocess_data=False)

# Creating the training image pairs and the corrsponding labels
if as_triplet:
    x_train, y_train = generate_image_triplets(train_images, train_labels)
    
    # Shuffling the training data
    np.random.seed(42)
    np.random.shuffle(x_train)

    x_train = [x_train[:, 0], x_train[:, 1], x_train[:, 2]]
else:
    x_train, y_train = generate_image_pairs(train_images, train_labels)
    
    # Shuffling the training data
    np.random.seed(42)
    np.random.shuffle(x_train)
    np.random.seed(42)
    np.random.shuffle(y_train)

    x_train = [x_train[:, 0], x_train[:, 1]]

# ------------------------------- CREATING AND COMPILING THE MODEL ------------------------------- #

if as_triplet:
    embedded_model = MobileNet_WithTop(
                        (image_height, image_width, 3), 
                        width_multiplier, depth_multiplier, 
                        dropout_rate,
                        data_augmentation,
                        use_weights,
                        frozen_layers_percent,
                        as_triplet=as_triplet
                    )
    siamese_model = SiameseModel_Triplet(embedded_model, (image_height, image_width, 3))
else:
    siamese_model = createSiameseModel_mobilenet(
                        (image_height, image_width, 3), 
                        width_multiplier, depth_multiplier, 
                        dropout_rate,
                        data_augmentation,
                        use_weights,
                        frozen_layers_percent,
                        as_triplet=as_triplet
                    )

siamese_model.summary()

siamese_model.compile(
            loss=losses_dict[loss],
            optimizer=optimizer,
            metrics=["accuracy"]
            )


# ------------------------------- TRAINING THE MODEL ------------------------------- #

history = siamese_model.fit(
            x_train,
            y_train,
            validation_split=validation_split,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks
            )


# ------------------------------- SAVING THE MODEL ------------------------------- #

if as_triplet:
    embedded_model.save_weights(savedModelPath + f'{model_name}.h5')
else:
    siamese_model.save_weights(savedModelPath + f'{model_name}.h5')