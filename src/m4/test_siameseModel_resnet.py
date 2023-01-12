# ------------------------------- IMPORTS ------------------------------- #

import matplotlib.pyplot as plt
from loadData import createDataset, generate_image_pairs, get_label, load_img

from create_siameseModel import createSiameseModel_resnet, contrastive_loss_with_margin, contrastive_loss_with_margin_alt
from test_functions import compute_accuracy, export_id_results_to_CSV, export_similarity_results_to_CSV, get_img_similarity_prediction

# ------------------------------- PARAMETERS ------------------------------- #

model_name = 'siamese_model_resnet_weights'
model_path = f'./log/saved_models/{model_name}/'
model_weights_path = f'./log/cps/{model_name}/{model_name}'
test_dir = './data/m4_manyOne/testKnownShort/'
batch_size = 4
image_height = 160
image_width = 160

margin = 1.0
weighted = True

# ------------------------------- MODEl EVALUATION ON TEST DATA ------------------------------- #

siamese_model = createSiameseModel_resnet((image_height, image_width, 3), 0.3, False, weighted)

siamese_model.compile(loss=contrastive_loss_with_margin_alt(margin=margin), optimizer='RMSprop', metrics=['accuracy'])
#siamese_model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
siamese_model.load_weights(model_path + f'{model_name}.h5')

x, y = createDataset(test_dir, (image_height, image_width))
x_pairs, y_pairs = generate_image_pairs(x, y)

preds = siamese_model.predict([x_pairs[:,0], x_pairs[:,1]])
loss = siamese_model.evaluate([x_pairs[:,0], x_pairs[:,1]], y_pairs)

test_accuracy = compute_accuracy(y_pairs[:], preds)

print(f'Loss: {loss}')
print(f'Predictions Accuracy: {test_accuracy}')

# ------------------------------- EXPORTING MODEL PREDICTIONS ON TEST DATA ------------------------------- #

export_id_results_to_CSV(siamese_model, model_path, test_dir, (image_height, image_width))
export_similarity_results_to_CSV(siamese_model, model_path, test_dir, (image_height, image_width))

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

pred = round(get_img_similarity_prediction(siamese_model, img1, img2), 4)
print(f'Similarity: {pred}%')

# Showing the image with the similarity prediction
ax = plt.subplot(1, 2, 1)
plt.imshow(img1)
ax = plt.subplot(1, 2, 2)
plt.imshow(img2)

plt.title(pred)