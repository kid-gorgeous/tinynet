import sys

sys.path.append('/Users/evan/tinynet')

from load_datasets import load_cars_dataset
train_generator, validation_generator = load_cars_dataset()

import keras
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D

import numpy as np
import scipy.io as sio

class model():
    def __init__(self):
        self.model = Sequential()
        self.model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
        self.model.add(Dense(196, activation='softmax', kernel_initializer='he_normal'))
        self.model.layers[0].trainable = False
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()

    def item_label(self):
        mat_data = sio.loadmat('/Users/evan/tinynet/datasets/stanford_cars/cars_annons.mat')
        self.item_label = mat_data['annotations']
        return 1

    def train(self, train_generator, validation_generator, epochs=1):
        self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator)

    def predict(self, image):
        return self.model.predict(image)

resnet50 = model()
resnet50.train(train_generator, validation_generator, epochs=1)
# Very Small application of visual training from the ResNet50 model
# bes accuracy is 0.0032 - val_loss: 11156
# best time : 318 seconds
model.save('resnet50.keras')



