# ------------------------------- IMPORTS ------------------------------- #

import tensorflow as tf
from tensorflow import keras
from loadData import createDataset, createDataframe, generate_image_pairs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from create_siameseModel import createSiameseModel_fromScratch

# ------------------------------- PARAMETERS ------------------------------- #

model_name = 'siamese_model_fromScratch'
model_path = f'./log/saved_models/{model_name}'
model_weights_path = f'./log/cps/{model_name}/{model_name}'
test_dir = './data/m4/test'
batch_size = 4
image_height = 160
image_width = 160

# ------------------------------- FUNCTIONS ------------------------------- #

def create_img_batch(img_path):
    # Loading and preprocessing the image
    img = tf.keras.utils.load_img(
        img_path, target_size=(image_height, image_width), color_mode='grayscale'
    )

    img_array = tf.keras.utils.img_to_array(img)
    img_array_batch = tf.expand_dims(img_array, 0) # Create a batch

    return img_array_batch

def get_img_predictions(model, img_paths):
    img1_array_batch = create_img_batch(img_paths[0])
    img2_array_batch = create_img_batch(img_paths[1])

    # Let the model make a prediction for the image
    preds = model.predict([img1_array_batch, img2_array_batch])

    # Getting a prediction for the image similarity percentage
    pred = round(preds[0][0], 4)

    return pred

def get_img_prediction_asID(model, img_path, test_imgs, test_labels):
    unique_labels = np.unique(test_labels)
    similarity_dict = {}

    for label in unique_labels:
        similarity_dict.setdefault(label, 0)

    img_array_batch = create_img_batch(img_path)

    for i, cmp_img in enumerate(test_imgs):
        cmp_img_array = tf.keras.utils.img_to_array(cmp_img)
        cmp_img_array_batch = tf.expand_dims(cmp_img_array, 0) # Create a batch

        pred = model.predict([img_array_batch, cmp_img_array_batch])
        # Getting a prediction for the image similarity percentage
        pred = round(pred[0][0], 4)

        similarity_dict[test_labels[i]] += pred

    return max(similarity_dict, key=similarity_dict.get)

def export_results_to_CSV(model, data_frame, x, y):
    df_length = len (data_frame.index)

    df_test = pd.DataFrame()
    empty_arr = [None] * df_length
    df_test['pred'] = empty_arr

    c = 0

    for img_path in data_frame['image']:
        pred = get_img_prediction_asID(model, img_path, x, y)
        
        df_test['pred'][c] = pred

        c+=1

    df_combined = data_frame.join(df_test)
    df_combined.to_csv(f'{model_path}_eval_results.csv')

    print(df_combined)

# ------------------------------- MODEl EVALUATION ON TEST DATA ------------------------------- #

siamese_model = createSiameseModel_fromScratch((image_height, image_width, 1), False)
siamese_model.compile(loss='binary_crossentropy', optimizer='adam', metrics='binary_accuracy')
siamese_model.load_weights(model_weights_path + ".h5")

x, y = createDataset(test_dir, (image_height, image_width), True)
x_pairs, y_pairs = generate_image_pairs(x, y)

results = siamese_model.evaluate([x_pairs[:,0], x_pairs[:,1]], y_pairs[:])

print(f'Loss: {results[0]}; Accuracy: {results[1]}')

# ------------------------------- MODEl PREDICTIONS ON TEST DATA ------------------------------- #

df = createDataframe(test_dir)
export_results_to_CSV(siamese_model, df, x, y)

# ------------------------------- MODEl PREDICTION ------------------------------- #

# Getting the image path for image to predict
img_paths = ('./data/m4/test/1_1.jpg', './data/m4/test/1_1.jpg')
img_path_split = img_paths[0].split('/')
img_name = img_path_split[len(img_path_split)-1]
img_name_split = img_name.split('_')

# Getting the actual age from the file name
if(len(img_name_split) > 1 and str.isnumeric(img_name_split[0])):
    actual = img_name_split[0]
else:
    actual = '?'

pred = get_img_predictions(siamese_model, img_paths)
print(f'Similarity: {pred}')

# Showing the image with the corresponding predictions
# ax = plt.subplot(2, 1, 1)
# plt.imshow(img1)
# plt.title("Same Face: {:.2f}%)".format(pred * 100, actual))