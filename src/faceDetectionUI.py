savedModelsDir = '../log/saved_models/'

def loadAllModels():
    return []

def predictImageWithModel(imagePath, modelPath):
    return 'Prediction:\nThe probability that the image contains a face is 0%'

while(1):
    
    print('Which image would you like to classify?')
    imagePath = input('Path to image:')
    print('\nWhich model would you like to use?')
    modelPath = input('Path to model:')

    print(f'\n{predictImageWithModel(imagePath, modelPath)}\n')
