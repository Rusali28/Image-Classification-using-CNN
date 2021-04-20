''' This file calls all the methods declared

datagen: loading the data from the directories
model: defining the model architecture
train: training the model and saving it 
'''

from datagen import datagen
from cnn import model
from train import train

training_path = 'dataset/training_set'
test_path = 'dataset/test_set'

'''loading the data '''
training_set, validation_set, test_set = datagen(training_path, test_path)

''' defining the model architecture '''
model = model()

''' training and savinf the model '''
train(model, 'model', training_set, validation_set, 32, 50)
