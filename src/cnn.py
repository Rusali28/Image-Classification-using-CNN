import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Dense, Flatten, MaxPooling2D

def model():

    ''' Defining the cnn model that performs the classification task.
    The model has 2 conv2D Layers each followed by a MaxPooling2D. The Flatten layer follows before the forward pass and classification task performed by the Fully Connected Layers with ReLU and Sigmoid function respectively'''
    
    model = Sequential()
    model.add(Convolution2D(32,3,3, input_shape = (64,64,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Convolution2D(32,3,3, activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    print(model.summary())

    return model