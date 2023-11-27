
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import getpass

user = getpass.getuser()
def load_datasets():
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # assuming images are in RGB format

    train_generator = train_datagen.flow_from_directory(
        '/Users/{}/tinynet/datasets/stanford_cars/cars_train'.format(user),  # replace with your actual path

        target_size= (150,150),
        batch_size=32,
        class_mode='categorical',
        subset='training')  # set as training data
    

    validation_generator = train_datagen.flow_from_directory(
        '/Users/{}/tinynet/datasets/stanford_cars/cars_train'.format(user),  # replace with your actual path
        
        target_size=(224, 224),  # resize images to this size
        batch_size=32,
        class_mode='categorical',
        subset='validation')  # set as validation data

    return train_generator, validation_generator

