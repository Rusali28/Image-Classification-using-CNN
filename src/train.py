from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy

def train(model, model_name, training_set, validation_set, batch_size, epochs, verbose = 1):
    
    ''' The 'model' will be trained using the Adam optimizer, with binary_crossentropy loss and accuracy as the metric on the 'training_set' using 'validation_set' for validation with the mentioned 'batch_size' for 'epoch' number of times '''

    model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['Accuracy'])
    
    model.fit(training_set, validation_data = validation_set, batch_size = batch_size, epochs = epochs, verbose = verbose)

    model.save('./models/'+model_name+'.h5')
