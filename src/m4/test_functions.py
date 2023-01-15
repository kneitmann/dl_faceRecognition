
# ------------------------------- IMPORTS ------------------------------- #

import tensorflow as tf
from tensorflow import keras
from loadData import createDataset, get_label, load_img, create_img_batch
import numpy as np
import pandas as pd
import os

# ------------------------------- UTILITY FUNCTIONS ------------------------------- #

def compute_triplet_accuracy(eval_model, x1, x2, y):
    """ Calculates the accuracy of the triplet model by comparing the predicted and actual similarity of the x1 and x2 images.

    Key arguments:
        eval_model -- The model to make the similarity predictions with.
        x1 -- Image 1 data
        x2 -- Image 2 data
        y -- Actual similarity of image 1 and image 2

    Returns:
        model_accuracy -- The mean accuracy of the model similarity predictions.
    """

    cos_sim = -(keras.losses.cosine_similarity(eval_model(x1), eval_model(x2)).numpy().reshape(-1,1))
    accuracy = tf.reduce_mean(keras.metrics.binary_accuracy(y.reshape(-1,1), cos_sim, threshold=0)).numpy()
    return accuracy

# ------------------------------- PREDICTION FUNCTIONS ------------------------------- #

def get_img_similarity_prediction(model, img1, img2):
    """ Makes a similarity predictions of two images with the given model.

    Key arguments:
        eval_model -- The model to make the similarity predictions with.
        x1 -- Image 1 data
        x2 -- Image 2 data

    Returns:
        similarity_prediction
    """

    img1_batch = create_img_batch(img1)
    img2_batch = create_img_batch(img2)

    # Let the model make a prediction for the image
    preds = model.predict([img1_batch, img2_batch])

    # Getting a prediction for the image similarity percentage
    pred = round(preds[0][0], 4)

    return pred

def get_tripletloss_similarity_predictions(model, x1, x2):
    """ Makes a similarity predictions of two images with the given model, using the cosine similarity.

    Key arguments:
        eval_model -- The model to make the similarity predictions with.
        x1 -- Image 1 data
        x2 -- Image 2 data

    Returns:
        similarity_prediction
    """

    # Let the model make a prediction for the image
    cos_sim = keras.losses.cosine_similarity(model(x1), model(x2)).numpy().reshape(-1,1)

    # Getting a prediction for the image similarity percentage
    pred = (-cos_sim) # Squeeze the cosine similarity in a range from 0 to 1

    return pred

def get_img_prediction_asID(model, img, test_imgs, test_labels, as_triplet=False):
    """ Makes a person ID predictions of an image with the given model.
        The chosen image gets a similarity score with each existing person ID in the test dataset.

    Key arguments:
        model -- The model to make the id predictions with.
        img -- The image to make a prediction for.
        test_imgs -- The test dataset
        test_labels -- The labels of the images in the test dataset
        as_triplet -- Indication, whether to make the prediction with a triplet model.

    Returns:
        id_prediction -- The predicted label (person ID) for the image.
    """

    # Creating a dictionary of all existing labels to save the predictions in
    unique_labels = np.unique(test_labels)
    similarity_dict = {}
    labels_count = {}

    for label in unique_labels:
        similarity_dict.setdefault(label, 0)        
        labels_count.setdefault(label, 0)

    # Pairing the image to compare with all other test images
    img_pairs = []

    for i, cmp_img in enumerate(test_imgs):
        img_pairs.append((img, cmp_img))

    img_pairs_nparr = np.array(img_pairs)

    # Making the predictions of all image pairs
    if as_triplet:
        preds = get_tripletloss_similarity_predictions(model, img_pairs_nparr[:, 0], img_pairs_nparr[:, 1])
    else:
        preds = model.predict([img_pairs_nparr[:, 0], img_pairs_nparr[:, 1]])
  
    for i, label in enumerate(test_labels):
        # Getting a prediction for the image similarity percentage 
        # and adding the value to the similarity dictionary
        pred = round(preds[i][0], 4)
        similarity_dict[label] += pred
        labels_count[label] += 1

    # Normalize similarity dictionary
    for label in unique_labels:
        similarity_dict[label] /= labels_count[label]

    # Returns the label with the highest similarity (lowest difference) score
    return min(similarity_dict, key=similarity_dict.get)

# ------------------------------- RESULTS EXPORT FUNCTIONS ------------------------------- #

def export_similarity_results_to_CSV(model, model_path, test_dir, img_size, grayscale=False, as_triplet=False):
    """ This function makes a similarity prediction on all possible image combinations of all given images in the test directory.
        The results will be saved in a data frame and exported as a CSV file.

    Key arguments:
        model -- The model to evaluate
        model_path -- The path where the model is saved (Results are saved in the directory)
        test_dir -- The directory path of the test dataset
        img_size -- The target image size
        grayscale -- Indication whether the images should be loaded in grayscale (default: False) 
        as_triplet -- Indication whether the model is a Triplet Loss model (default: False)

    """

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

                if as_triplet:               
                    x1_batch = create_img_batch(img1)
                    x2_batch = create_img_batch(img2)
                    pred = 1-get_tripletloss_similarity_predictions(model, x1_batch, x2_batch)[0][0]
                else:
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

def export_id_results_to_CSV(model, model_path, test_dir, img_size, grayscale=False, as_triplet=False):
    """ This function makes a person ID prediction for each image given in the test directory.
        The results will be saved in a data frame and exported as a CSV file.

    Key arguments:
        model -- The model to evaluate
        model_path -- The path where the model is saved (Results are saved in the directory)
        test_dir -- The directory path of the test dataset
        img_size -- The target image size
        grayscale -- Indication whether the images should be loaded in grayscale (default: False) 
        as_triplet -- Indication whether the model is a Triplet Loss model (default: False)

    """
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
            pred = get_img_prediction_asID(model, img, test_x, test_y, as_triplet=as_triplet)
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