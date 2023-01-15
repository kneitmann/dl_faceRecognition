# ------------------------------- IMPORTS ------------------------------- #

import matplotlib.pyplot as plt
from loadData import createDataset, generate_image_pairs, get_label, load_img, generate_image_triplets

from create_siameseModel import createSiameseModel_mobilenet, contrastive_loss_with_margin, triplet_loss, MobileNet_WithTop
from test_functions import *

# ------------------------------- PARAMETERS ------------------------------- #

model_name = 'siamese_model_mobilenet_weights_margin0,5_frozen1,0_triplet'
model_path = f'./log/saved_models/{model_name}/'
model_weights_path = f'./log/cps/{model_name}/{model_name}'
test_dir = './data/m4_manyOne/testKnownShort/'
batch_size = 4
image_height = 128
image_width = 128

loss = 'triplet_loss'
as_triplet = (loss=='triplet_loss')
weighted = True
frozen_layers_percent = 1.0
margin = 0.5

losses_dict = {
    'binary_crossentropy' : 'binary_crossentropy',
    'contrastive_loss' : contrastive_loss_with_margin(margin=margin),
    'triplet_loss' : triplet_loss(margin=margin)
}

# ------------------------------- MODEl EVALUATION ON TEST DATA ------------------------------- #

# Creating the test dataset
test_images, test_labels = createDataset(test_dir, (image_height, image_width), preprocess_data=False)

# Creating the test image pairs and the corrsponding labels
x_test, y_test = generate_image_pairs(test_images, test_labels)

if as_triplet:
    model = MobileNet_WithTop(
                (image_height, image_width, 3),
                1, 1,
                0.3,
                False,
                weighted,
                frozen_layers_percent,
                as_triplet=as_triplet
                )
    model.load_weights(model_path + f'{model_name}.h5')
    accuracy = compute_triplet_accuracy(model, x_test[:, 0], x_test[:, 1], y_test)

    print(f'Accuracy: {accuracy}')

else:
    model = createSiameseModel_mobilenet(
                (image_height, image_width, 3), 
                1, 1,
                0.3,
                False,
                weighted,
                frozen_layers_percent,
                as_triplet=(loss=='triplet_loss')
                )
                        
    model.load_weights(model_path + f'{model_name}.h5')
    model.compile(loss=losses_dict[loss], optimizer='RMSProp', metrics=['accuracy'])

    preds = model.predict([x_test[:, 0], x_test[:, 1]])
    results = model.evaluate([x_test[:, 0], x_test[:, 1]], y_test)

    print(f'Loss: {results[0]}; Accuracy: {results[1]}')

# ------------------------------- EXPORTING MODEL PREDICTIONS ON TEST DATA ------------------------------- #

export_id_results_to_CSV(model, model_path, test_dir, (image_height, image_width), as_triplet=as_triplet)
# export_similarity_results_to_CSV(model, model_path, test_dir, (image_height, image_width), as_triplet=as_triplet)

# ------------------------------- MODEL SINGLE PREDICTION ------------------------------- #

# Getting the image path for image to predict
img_paths = ('./data/m4/test/1_1.jpg', './data/m4/test/7_14.jpg')

img_path1_split = img_paths[0].split('/')
img_path2_split = img_paths[1].split('/')
img1_name = img_path1_split[len(img_path1_split)-1]
img2_name = img_path1_split[len(img_path2_split)-1]

# Getting the actual id from the file name
person1_id = get_label(img1_name)
person2_id = get_label(img2_name)

img1 = load_img(img_paths[0], (image_height, image_width), grayscale=False, preprocess_img=True)
img2 = load_img(img_paths[1], (image_height, image_width), grayscale=False, preprocess_img=True)

if as_triplet:
    pred = round(get_tripletloss_similarity_predictions(model, img1, img2), 4)
else:
    pred = round(get_img_similarity_prediction(model, img1, img2), 4)

print(f'Similarity: {pred}%')

# Showing the image with the similarity prediction
ax = plt.subplot(1, 2, 1)
plt.imshow(img1)
ax = plt.subplot(1, 2, 2)
plt.imshow(img2)

plt.title(pred)