# ------------------------------- IMPORTS ------------------------------- #

import tensorflow as tf
from tensorflow import keras
from loadData import createDataset, createDataframe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------- PARAMETERS ------------------------------- #

model_path = './log/saved_models/siamese_model/'
test_dir = './data/m4/test'
batch_size = 8
image_height = 160
image_width = 160

# ------------------------------- FUNCTIONS ------------------------------- #

def get_img_predictions(img_paths):
    # Loading and preprocessing the image
    img1 = tf.keras.utils.load_img(
        img_paths[0], target_size=(image_height, image_width)
    )
    img2 = tf.keras.utils.load_img(
        img_paths[1], target_size=(image_height, image_width)
    )

    img1_array = tf.keras.utils.img_to_array(img1)
    img1_array_batch = tf.expand_dims(img1_array, 0) # Create a batch
    img2_array = tf.keras.utils.img_to_array(img2)
    img2_array_batch = tf.expand_dims(img2_array, 0) # Create a batch

    # Let the model make a prediction for the image
    preds = loaded_model.predict([img1_array_batch, img2_array_batch])

    # Getting face, mask and age prediction
    pred = round(preds[0][0][0], 4)

    return img, pred

def export_results_to_CSV(img_path):
    df = createDataframe(test_dir)
    df_length = len (df.index)

    df_test = pd.DataFrame()
    empty_arr = [None] * df_length
    df_test['pred'] = empty_arr
    df_test['pred_diff'] = empty_arr

    c = 0

    for img_path in df['image']:
        img, pred = get_img_predictions(img_path)
        
        df_test['pred'][c] = pred

        df_test['pred_diff'][c] = round(abs(df['face'][c] - pred), 4)

        c+=1

    df_combined = df.join(df_test)
    df_combined.to_csv(f'{model_path}eval_results.csv')

    print(df_combined)

# ------------------------------- MODEl EVALUATION ON TEST DATA ------------------------------- #

loaded_model = keras.models.load_model(model_path)

x_pairs, y_pairs = createDataset(test_dir, (image_height, image_width))

results = loaded_model.evaluate([x_pairs[:,0], x_pairs[:,1]], y_pairs[:])

print(f'Loss: {results[0]}; Accuracy: {results[1]}')

# ------------------------------- MODEl PREDICTION ------------------------------- #

# Getting the image path for image to predict
img_paths = ('../../data/m3/extras/hidden-face-boy-dp.jpg', '../../data/m3/extras/hidden-face-boy-dp.jpg')
img_path_split = img_paths[0].split('/')
img_name = img_path_split[len(img_path_split)-1]
img_name_split = img_name.split('_')

# Getting the actual age from the file name
if(len(img_name_split) > 1 and str.isnumeric(img_name_split[0])):
    actual = img_name_split[0]
else:
    actual = '?'

img, pred = get_img_predictions(img_paths)

# Showing the image with the corresponding predictions
ax = plt.subplot(1, 1, 1)
plt.imshow(img)
plt.title("Face: {:.2f}% | Mask: {:.2f}% | Age: {} (Actual: {})".format(pred * 100, actual))