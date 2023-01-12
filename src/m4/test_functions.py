
# ------------------------------- IMPORTS ------------------------------- #

import tensorflow as tf
from tensorflow import keras
from loadData import createDataset, createDataframe, generate_image_pairs, get_label, load_img, create_img_batch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# ------------------------------- UTILITY FUNCTIONS ------------------------------- #

def compute_accuracy(y_true, y_pred):
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)

# ------------------------------- PREDICTION FUNCTIONS ------------------------------- #

def get_img_similarity_prediction(model, img1, img2):
    img1_batch = create_img_batch(img1)
    img2_batch = create_img_batch(img2)

    # Let the model make a prediction for the image
    preds = model.predict([img1_batch, img2_batch])

    # Getting a prediction for the image similarity percentage
    pred = round(preds[0][0], 4)

    return pred

def get_img_prediction_asID(model, img, test_imgs, test_labels):
    # Creating a dictionary of all existing labels to save the predictions in
    unique_labels = np.unique(test_labels)
    similarity_dict = {}

    for label in unique_labels:
        similarity_dict.setdefault(label, 0)

    # Pairing the image to compare with all other test images
    img_pairs = []

    for i, cmp_img in enumerate(test_imgs):
        img_pairs.append((img, cmp_img))

    img_pairs_nparr = np.array(img_pairs)
    # Making the predictions of all image pairs
    preds = model.predict([img_pairs_nparr[:, 0], img_pairs_nparr[:, 1]])
  
    for i, label in enumerate(test_labels):
        # Getting a prediction for the image similarity percentage 
        # and adding the value to the similarity dictionary
        pred = 1-round(preds[i][0], 4)
        similarity_dict[label] += pred

    # Returns the label with the highest similarity score
    return max(similarity_dict, key=similarity_dict.get)

# ------------------------------- RESULTS EXPORT FUNCTIONS ------------------------------- #

### This function makes a similarity prediction on all possible image combinations of all given images in the test directory. ###
### The results will be saved in a data frame and exported as a CSV file. ###
def export_similarity_results_to_CSV(model, model_path, test_dir, img_size, grayscale=False):
    img_names = []
    img_labels = []
    cmp_img_names = []
    cmp_img_labels = []
    preds = []
    preds_round = []
    actuals = []
    pred_diffs = []
    pred_diffs_round = []

    # Making the similarity prediction for every possible image combination
    for dirpath, dirs, files in os.walk(test_dir): 
        for i, filename in enumerate(files):
            for j, cmp_filename in enumerate(files[i:len(files)]):
                img_path = os.path.join(dirpath, filename)
                cmp_img_path = os.path.join(dirpath, cmp_filename)

                img_names.append(filename)
                cmp_img_names.append(cmp_filename)

                label = get_label(filename)
                img_labels.append(label)
                cmp_label = get_label(cmp_filename)
                cmp_img_labels.append(cmp_label)

                img1 = load_img(img_path, img_size, grayscale, False)
                img2 = load_img(cmp_img_path, img_size, grayscale, False)

                pred = 1-get_img_similarity_prediction(model, img1, img2)
                preds.append(pred)
                preds_round.append(round(pred, 0))
                actual = int(label==cmp_label)
                actuals.append(actual)
                pred_diff = round(abs(actual-pred), 4)
                pred_diffs.append(pred_diff)
                pred_diffs_round.append(round(pred_diff, 0))

    # Creating the data frame
    df = pd.DataFrame()

    df['image'] = img_names # Image file name
    df['cmp_image'] = cmp_img_names # Compared image file name
    df['image_label'] = img_labels # Image label (person ID)
    df['cmp_image_label'] = cmp_img_labels # Compared image label (person ID)
    df['actual'] = actuals # Actual similarity value (0 or 1)
    df['pred'] = preds # Similarity prediction
    df['pred_round'] = preds_round # Rounded similarity prediction (0 or 1)
    df['pred_diff'] = pred_diffs # (Absolute) Difference of prediction vs the actual value
    df['pred_diff_round'] = pred_diffs_round # Rounded difference of prediction vs the actual value (0 or 1)

    # Exporting the data frame as a CSV file
    df.to_csv(f'{model_path}eval_similarity_results.csv')

    print(df)

### This function makes a person ID prediction for each image given in the test directory. ###
### The results will be saved in a data frame and exported as a CSV file. ###
def export_id_results_to_CSV(model, model_path, test_dir, img_size, grayscale=False):
    img_names = []
    img_labels = []
    preds = []
    pred_diffs = []

    test_x, test_y = createDataset(test_dir, img_size, grayscale=grayscale)

    # Making a prediction for the person ID for each image in the test dataset
    for dirpath, dirs, files in os.walk(test_dir): 
        for filename in files:
            img_path = os.path.join(dirpath, filename)

            img_names.append(filename)
            label = get_label(filename)
            img_labels.append(label)

            img = load_img(img_path, img_size, grayscale)
            pred = get_img_prediction_asID(model, img, test_x, test_y)
            preds.append(pred)
            pred_diffs.append(int(pred!=label))

    # Creating the data frame
    df = pd.DataFrame()

    df['image'] = img_names # Image file name
    df['image_label'] = img_labels # Image label (person ID)
    df['pred'] = preds # Label prediction
    df['pred_diff'] = pred_diffs # Difference of prediction vs the actual value (0 or 1)

    # Exporting the data frame as a CSV file
    df.to_csv(f'{model_path}eval_id_results.csv')

    print(df)