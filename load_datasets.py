from termcolor import colored as c
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import pandas as pd
import getpass 
import cv2

user = getpass.getuser()


# Class that will prepare the data for the Application
class Datasets:
    def __init__(self):
        pass

    def load_geolocations(self, withExtra):   
        if withExtra == True:
            # data from the Dr5hn Country City State API from Kaggle.com
            geoloc = '/Users/evan/tinynet/datasets/locations/extras/cities.csv'
            geo_df = pd.read_csv(geoloc)
        else:
            # data from the Dr5hn Country City State API
            geoloc = '/Users/evan/tinynet/datasets/locations/csv/cities.csv'
            geo_df = pd.read_csv(geoloc)
        return geo_df

    def load_lidar(self):
        print(c("Loading Lidar Data" ,"green"))

        wdir = '/Users/evan/tinynet/datasets/lidar2'

        print(c(wdir, 'green'))


    def load_roads_dataset(self):
        metadata = pd.read_csv('/Users/evan/tinynet/datasets/roads/metadata.csv')
        print(metadata.filename)
        filename = '/Images/Images/clean_1.jpg'
        dir = "/Users/{}/tinynet/datasets/roads".format(user)

        return data

    def load_cars_dataset(self, image_size=224):
        train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # assuming images are in RGB format
        train_generator = train_datagen.flow_from_directory(
            '/Users/{}/tinynet/datasets/stanford_cars/cars_train'.format(user),  # replace with your actual path
            target_size= (image_size,image_size),  # resize images to this size
            batch_size=32,
            class_mode='categorical',
            subset='training')  # set as training data
        
        validation_generator = train_datagen.flow_from_directory(
            '/Users/{}/tinynet/datasets/stanford_cars/cars_train'.format(user),  # replace with your actual path
            target_size=(image_size,image_size),  # resize images to this size
            batch_size=32,
            class_mode='categorical',
            subset='validation')  # set as validation data

        print("Dataset Loaded")
        return train_generator, validation_generator


# if __name__ == "__main__":
    
#     data = Datasets()
#     locations = data.load_geolocations(True)
    
#     data.load_lidar()

