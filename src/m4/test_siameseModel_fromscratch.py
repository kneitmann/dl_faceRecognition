# ------------------------------- IMPORTS ------------------------------- #

from loadData import createDataset, createDataframe, generate_image_pairs

from create_siameseModel import createSiameseModel_fromScratch, contrastive_loss_with_margin
from test_functions import export_similarity_results_to_CSV, export_id_results_to_CSV, get_img_similarity_prediction, compute_accuracy

# ------------------------------- PARAMETERS ------------------------------- #

model_name = 'siamese_model_fromScratch'
model_path = f'./log/saved_models/{model_name}/'
model_weights_path = f'./log/cps/{model_name}/{model_name}'
test_dir = './data/m4_manyOne/testKnownShort/'
batch_size = 4
image_height = 160
image_width = 160

# ------------------------------- MODEl EVALUATION ON TEST DATA ------------------------------- #

siamese_model = createSiameseModel_fromScratch((image_height, image_width, 1), False)
siamese_model.compile(loss=contrastive_loss_with_margin, optimizer='adam')
siamese_model.load_weights(model_path + f'{model_name}.h5')

x, y = createDataset(test_dir, (image_height, image_width), grayscale=True)
x_pairs, y_pairs = generate_image_pairs(x, y)

results = siamese_model.predict([x_pairs[:,0], x_pairs[:,1]])
test_accuracy = compute_accuracy(y_pairs[:], results)

#print(f'Loss: {results[0]}; Accuracy: {results[1]}')
print(f'Accuracy: {test_accuracy}')

# ------------------------------- EXPORTING MODEL PREDICTIONS ON TEST DATA ------------------------------- #

export_id_results_to_CSV(siamese_model, model_path, test_dir, (image_height, image_width), grayscale=True)
export_similarity_results_to_CSV(siamese_model, model_path, test_dir, (image_height, image_width), grayscale=True)

# ------------------------------- MODEl PREDICTION ------------------------------- #

# Getting the image path for image to predict
img_paths = ('./data/m4/test/1_1.jpg', './data/m4/test/1_2.jpg')
img_path_split = img_paths[0].split('/')
img_name = img_path_split[len(img_path_split)-1]
img_name_split = img_name.split('_')

# Getting the actual age from the file name
if(len(img_name_split) > 1 and str.isnumeric(img_name_split[0])):
    actual = img_name_split[0]
else:
    actual = '?'

pred = get_img_similarity_prediction(siamese_model, img_paths, (image_height, image_width))
print(f'Similarity: {pred}%')

# Showing the image with the corresponding predictions
# ax = plt.subplot(2, 1, 1)
# plt.imshow(img1)
# plt.title("Same Face: {:.2f}%)".format(pred * 100, actual))