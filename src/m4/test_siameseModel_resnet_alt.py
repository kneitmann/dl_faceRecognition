# ------------------------------- IMPORTS ------------------------------- #

import matplotlib.pyplot as plt
from loadData import createDataset, generate_image_pairs, get_label, load_img

from create_siameseModel import createSiameseModel_resnet_alt, contrastive_loss_with_margin
from test_functions import export_similarity_results_to_CSV, export_id_results_to_CSV, get_img_similarity_prediction, compute_accuracy

# ------------------------------- PARAMETERS ------------------------------- #

model_name = 'siamese_model_mobilenet_alt'
model_path = f'./log/saved_models/{model_name}/'
model_weights_path = f'./log/cps/{model_name}/'
test_dir = './data/m4_manyOne/testKnownShort/'
batch_size = 4
image_height = 160
image_width = 160

# ------------------------------- MODEl EVALUATION ON TEST DATA ------------------------------- #

siamese_model = createSiameseModel_resnet_alt((image_height, image_width, 3), 1, 1, 0.3, False)
siamese_model.compile(loss=contrastive_loss_with_margin, optimizer='adam')
siamese_model.load_weights(model_path + f'{model_name}.h5')

x, y = createDataset(test_dir, (image_height, image_width))
x_pairs, y_pairs = generate_image_pairs(x, y)

preds = siamese_model.predict([x_pairs[:,0], x_pairs[:,1]])
results = siamese_model.evaluate([x_pairs[:,0], x_pairs[:,1]], y_pairs)

print(f'Loss: {results[0]}; Accuracy: {results[1]}')

# ------------------------------- EXPORTING MODEL PREDICTIONS ON TEST DATA ------------------------------- #

export_similarity_results_to_CSV(siamese_model, model_path, test_dir, (image_height, image_width))
export_id_results_to_CSV(siamese_model, model_path, test_dir, (image_height, image_width))

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

pred = get_img_similarity_prediction(siamese_model, img1, img2)
print(f'Similarity: {pred}%')

# Showing the image with the similarity prediction
ax = plt.subplot(1, 2, 1)
plt.imshow(img1)
ax = plt.subplot(1, 2, 2)
plt.imshow(img2)

plt.title(pred)