
import tensorflow as tf
from tensorflow import keras
from loadData import createDataset, createDataframe, generate_image_pairs, get_label, load_img
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def create_img_batch(img_path, image_size, grayscale=False):
    color_mode = 'rgb'
    if grayscale==True: color_mode = 'grayscale'

    # Loading and preprocessing the image
    img = tf.keras.utils.load_img(
        img_path, target_size=image_size, color_mode=color_mode
    )

    img_array = tf.keras.utils.img_to_array(img)
    img_array_batch = tf.expand_dims(img_array, 0) # Create a batch

    return img_array_batch

def compute_accuracy(y_true, y_pred):
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)

def get_img_similarity_prediction(model, img1, img2):
    # Let the model make a prediction for the image
    preds = model.predict([img1, img2])

    # Getting a prediction for the image similarity percentage
    pred = round(preds[0][0], 4)

    return 1-pred

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

    img_pairs = np.array(img_pairs)
    # Making the predictions of all image pairs
    preds = model.predict([img_pairs[:, 0], img_pairs[:, 1]])
  
    for i, label in enumerate(test_labels):
        # Getting a prediction for the image similarity percentage 
        # and adding the value to the similarity dictionary
        pred = 1-round(preds[i][0], 4)
        similarity_dict[label] += pred

    # Returns the label with the highest similarity score
    return max(similarity_dict, key=similarity_dict.get)

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
                label = get_label(filename)
                img_labels.append(label)

                cmp_img_names.append(cmp_filename)
                cmp_label = get_label(cmp_filename)
                cmp_img_labels.append(cmp_label)

                img1 = create_img_batch(img_path, img_size, grayscale)
                img2 = create_img_batch(cmp_img_path, img_size, grayscale)

                pred = get_img_similarity_prediction(model, img1, img2)
                preds.append(pred)
                preds_round.append(round(pred, 0))
                actual = int(label==cmp_label)
                actuals.append(actual)
                pred_diff = round(abs(actual-pred), 4)
                pred_diffs.append(pred_diff)
                pred_diffs_round.append(round(pred_diff, 0))

    # Creating the data frame
    df = pd.DataFrame()

    df['image'] = img_names
    df['image_label'] = img_labels
    df['cmp_image'] = cmp_img_names
    df['cmp_image_label'] = cmp_img_labels
    df['actual'] = actuals
    df['pred'] = preds
    df['pred_round'] = preds_round
    df['pred_diff'] = pred_diffs
    df['pred_diff_round'] = pred_diffs_round

    # Exporting the data frame as a CSV file
    df.to_csv(f'{model_path}eval_similarity_results.csv')

    print(df)

def export_id_results_to_CSV(model, model_path, test_dir, img_size, grayscale=False):
    img_names = []
    img_labels = []
    preds = []
    pred_diffs = []

    test_x, test_y = createDataset(test_dir, img_size)

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
            pred_diffs.append(int(pred==label))

    # Creating the data frame
    df = pd.DataFrame()

    df['image'] = img_names
    df['image_label'] = img_labels
    df['pred'] = preds
    df['pred_diff'] = pred_diffs

    # Exporting the data frame as a CSV file
    df.to_csv(f'{model_path}eval_id_results.csv')

    print(df)