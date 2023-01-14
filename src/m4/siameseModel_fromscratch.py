# ------------------------------- IMPORTS ------------------------------- #

import os
import numpy as np

from tensorflow import keras

from loadData import createDataset, generate_image_pairs, generate_image_triplets
from create_siameseModel import createSiameseModel_fromScratch, contrastive_loss_with_margin_alt, triplet_loss

# ------------------------------- PARAMETERS ------------------------------- #

# Log parameters
model_name = 'siamese_model_fromScratch_margin0,75'
savedModelPath = f'./log/saved_models/{model_name}/'
tb_log_dir = f'./log/tensorboard/{model_name}/'
cp_filepath = f'./log/cps/{model_name}/'
training_data_path = './data/m4_manyOne/training/'
validation_data_path = './data/m4_manyOne/validation/'

if not os.path.exists(cp_filepath):
    os.makedirs(cp_filepath)

if not os.path.exists(savedModelPath):
    os.makedirs(savedModelPath)


# Hyperparameters
learningRate = 0.001
doDataAugmentation = False
dropoutRate = 0.3
width_multiplier = 1
depth_multiplier = 1

## Contrastive Loss parameters
margin = 0.75

## Triplet Loss parameters
emb_size = 128
alpha = 0.2

# Training parameters
batch_size = 32
epochs = 30
validation_split = 0.2
useWeights = True
decay = learningRate/epochs
frozen_layers_percent = 0.75
loss = 'triplet_loss'
optimizer = keras.optimizers.Adam(learningRate) if loss == 'triplet_loss' else keras.optimizers.RMSprop(learningRate)

# Data parameters
image_height = 128
image_width = 128

def lr_time_decay(epoch, lr):
    return lr*1/(1+decay*epoch)

losses_dict = {
    'binary_crossentropy' : 'binary_crossentropy',
    'contrastive_loss' : contrastive_loss_with_margin_alt(margin=margin),
    'triplet_loss' : triplet_loss(emb_size=emb_size, alpha=alpha)
}
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
            start_from_epoch=10,
            )
]

# ----------------------------------- DATASETS ----------------------------------- #

# Creating the training and validation dataset
train_images, train_labels = createDataset(training_data_path, (image_height, image_width), preprocess_data=(loss == 'triplet_loss'))

# Creating the training image pairs and the corrsponding labels
if loss == 'triplet_loss':
    x_train, y_train = generate_image_triplets(train_images, train_labels)
    
    # Shuffling the training data
    np.random.seed(42)
    np.random.shuffle(x_train)
    np.random.seed(42)
    np.random.shuffle(y_train)
else:
    x_train, y_train = generate_image_pairs(train_images, train_labels)
    
    # Shuffling the training data
    np.random.seed(42)
    np.random.shuffle(x_train)
    np.random.seed(42)
    np.random.shuffle(y_train)

    x_train = [x_train[:, 0], x_train[:, 1]]

# ------------------------------- CREATING AND COMPILING MODEL ------------------------------- #

siamese_model = createSiameseModel_fromScratch((image_height, image_width, 1), doDataAugmentation)
keras.utils.plot_model(siamese_model, to_file=f'siamese_model.png', show_layer_activations=True)
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

siamese_model.save_weights(savedModelPath + f'{model_name}.h5')