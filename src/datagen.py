''' Please download the dataset folder from the link given in README.md and save it in the src directory. One can send access request in order to use the same dataset. One can also choose any other dataset of their choice (please change the path mentioned in the code accordingly). The dataset contains 10,005 images of 2 classes viz Cat and Dog. The data is split into 8,005 training images and 2,000 testing images. '''

from keras.preprocessing.image import ImageDataGenerator  

def datagen():

    ''' Function for loading the dataset using Image Data Generator '''

    training_path = 'dataset/training_set'
    test_path = 'dataset/test_set'

    train_datagen = ImageDataGenerator(validation_split=0.2, rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True )

    val_datagen = ImageDataGenerator(validation_split=0.2,rescale = 1/255)

    test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_directory(training_path,
                                                target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'binary',
                                                subset = 'training')

    validation_set = val_datagen.flow_from_directory(training_path,
                                                target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'binary',
                                                subset = 'validation')


    test_set = test_datagen.flow_from_directory(test_path,
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

    return training_set, validation_set, test_set