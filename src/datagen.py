from keras.preprocessing.image import ImageDataGenerator

def datagen():
    train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True )
    test_datagen = ImageDataGenerator(rescale = 1./255)

    from google.colab import drive
    drive.mount("/content/gdrive")

    training_set = train_datagen.flow_from_directory('/content/gdrive/My Drive/dataset/training_set',
                                                target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'binary')

    test_set = test_datagen.flow_from_directory('/content/gdrive/My Drive/dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

    return train_datagen, test_datagen