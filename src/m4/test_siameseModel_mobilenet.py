# ------------------------------- IMPORTS ------------------------------- #

from loadData import createDataset, createDataframe, generate_image_pairs, get_label

from create_siameseModel import createSiameseModel_mobilenet, contrastive_loss_with_margin, contrastive_loss_with_margin2
from test_functions import compute_accuracy, export_id_results_to_CSV, export_similarity_results_to_CSV, get_img_similarity_prediction, create_img_batch

# ------------------------------- PARAMETERS ------------------------------- #

model_name = 'siamese_model_mobilenet'
model_path = f'./log/saved_models/{model_name}/'
model_weights_path = f'./log/cps/{model_name}/{model_name}'
test_dir = './data/m4_manyOne/testKnownShort/'
batch_size = 4
image_height = 160
image_width = 160

# ------------------------------- MODEl EVALUATION ON TEST DATA ------------------------------- #

siamese_model = createSiameseModel_mobilenet((image_height, image_width, 3), 1, 1, 0.3, False)
siamese_model.compile(loss=contrastive_loss_with_margin(margin=0.5), optimizer='adam')
siamese_model.load_weights(model_path + f'{model_name}.h5')

x, y = createDataset(test_dir, (image_height, image_width))
x_pairs, y_pairs = generate_image_pairs(x, y)

results = siamese_model.predict([x_pairs[:,0], x_pairs[:,1]])
test_accuracy = compute_accuracy(y_pairs[:], results)

#print(f'Loss: {results[0]}; Accuracy: {results[1]}')
print(f'Accuracy: {test_accuracy}')

# ------------------------------- EXPORTING MODEL PREDICTIONS ON TEST DATA ------------------------------- #

#export_id_results_to_CSV(siamese_model, model_path, test_dir, (image_height, image_width))
export_similarity_results_to_CSV(siamese_model, model_path, test_dir, (image_height, image_width))

# ------------------------------- MODEl SINGLE PREDICTION ------------------------------- #

# Getting the image path for image to predict
img_paths = ('./data/m4/test/1_1.jpg', './data/m4/test/1_2.jpg')

# Getting the actual id from the file name
person1_id = get_label(img_paths[0])
person2_id = get_label(img_paths[1])

img1 = create_img_batch(img_paths[0], (image_height, image_width))
img2 = create_img_batch(img_paths[1], (image_height, image_width))

pred = get_img_similarity_prediction(siamese_model, img1, img2)
print(f'Similarity: {pred}%')

# Showing the image with the corresponding predictions
# ax = plt.subplot(2, 1, 1)
# plt.imshow(img1)
# plt.title("Same Face: {:.2f}%)".format(pred * 100, actual))