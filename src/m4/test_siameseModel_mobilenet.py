# ------------------------------- IMPORTS ------------------------------- #

import matplotlib.pyplot as plt
from loadData import createDataset, createDataframe, generate_image_pairs, get_label, load_img

from create_siameseModel import createSiameseModel_mobilenet_weighted, createSiameseModel_mobilenet_noWeights, contrastive_loss_with_margin, contrastive_loss_with_margin2
from test_functions import compute_accuracy, export_id_results_to_CSV, export_similarity_results_to_CSV, get_img_similarity_prediction, create_img_batch

# ------------------------------- PARAMETERS ------------------------------- #

model_name = 'siamese_model_mobilenet'
model_path = f'./log/saved_models/{model_name}/'
model_weights_path = f'./log/cps/{model_name}/{model_name}'
test_dir = './data/m4_manyOne/testKnownShort/'
batch_size = 4
image_height = 160
image_width = 160

margin = 0.3
weighted = True

# ------------------------------- MODEl EVALUATION ON TEST DATA ------------------------------- #

if weighted:
    siamese_model = createSiameseModel_mobilenet_weighted((image_height, image_width, 3), 1, 1, 0.3, False)
else:
    siamese_model = createSiameseModel_mobilenet_noWeights((image_height, image_width, 3), 1, 1, 0.3, False)

siamese_model.compile(loss=contrastive_loss_with_margin(margin=margin), optimizer='adam')
siamese_model.load_weights(model_path + f'{model_name}.h5')

x, y = createDataset(test_dir, (image_height, image_width))
x_pairs, y_pairs = generate_image_pairs(x, y)

preds = siamese_model.predict([x_pairs[:,0], x_pairs[:,1]])
loss = siamese_model.evaluate([x_pairs[:,0], x_pairs[:,1]])

test_accuracy = compute_accuracy(y_pairs[:], preds)

print(f'Loss: {loss}')
print(f'Predictions Accuracy: {test_accuracy}')

# ------------------------------- EXPORTING MODEL PREDICTIONS ON TEST DATA ------------------------------- #

export_id_results_to_CSV(siamese_model, model_path, test_dir, (image_height, image_width))
export_similarity_results_to_CSV(siamese_model, model_path, test_dir, (image_height, image_width))

# ------------------------------- MODEl SINGLE PREDICTION ------------------------------- #

# Getting the image path for image to predict
img_paths = ('./data/m4/test/1_1.jpg', './data/m4/test/1_2.jpg')

img_path1_split = img_paths[0].split('/')
img_path2_split = img_paths[1].split('/')
img1_name = img_path1_split[len(img_path1_split)-1]
img2_name = img_path1_split[len(img_path2_split)-1]

# Getting the actual id from the file name
person1_id = get_label(img1_name)
person2_id = get_label(img2_name)

img1 = load_img(img_paths[0], (image_height, image_width), grayscale=False, preprocess_img=False)
img2 = load_img(img_paths[1], (image_height, image_width), grayscale=False, preprocess_img=False)
img1_batch = create_img_batch(img1)
img2_batch = create_img_batch(img2)

pred = get_img_similarity_prediction(siamese_model, img1_batch, img2_batch)
print(f'Similarity: {pred}%')

# Showing the image with the similarity prediction
ax = plt.subplot(1, 2, 1)
plt.imshow(img1)
ax = plt.subplot(1, 2, 2)
plt.imshow(img2)

plt.title(pred)