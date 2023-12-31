import sys
import time
import keras
import getpass
import tensorflow as tf

sys.path.append('/Users/{}/tinynet'.format(getpass.getuser()))

from keras.datasets import fashion_mnist

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
optimizer = 'adam'
loss = 'categorical_crossentropy'
metrics = ['accuracy']



# NN with Keras`
class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            keras.layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False),
            keras.layers.BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                keras.layers.Conv2D(filters, 1, strides=strides, padding="same", use_bias=False),
                keras.layers.BatchNormalization()
            ]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)


    
from keras.models import Sequential
class ResNet34(keras.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = keras.layers.Conv2D(64, 7, strides=2, input_shape=[224, 224, 3], padding="same", use_bias=False)
        self.bn1 = keras.layers.BatchNormalization()
        self.maxpool = keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")
        self.residual_layers = [
            ResidualUnit(64, strides=1),
            ResidualUnit(64, strides=1),
            ResidualUnit(64, strides=1),
            ResidualUnit(128, strides=2),
            ResidualUnit(128, strides=1),
            ResidualUnit(128, strides=1),
            ResidualUnit(128, strides=1),
            ResidualUnit(256, strides=2),
            ResidualUnit(256, strides=1),
            ResidualUnit(256, strides=1),
            ResidualUnit(256, strides=1),
            ResidualUnit(256, strides=1),
            ResidualUnit(256, strides=1),
            ResidualUnit(512, strides=2),
            ResidualUnit(512, strides=1),
            ResidualUnit(512, strides=1),
        ]
        self.avgpool = keras.layers.GlobalAvgPool2D()
        self.fc = keras.layers.Dense(output_dim, activation="softmax")
        
    def call(self, inputs):
        Z = self.hidden1(inputs)
        Z = self.bn1(Z)
        Z = tf.nn.relu(Z)
        Z = self.maxpool(Z)
        for layer in self.residual_layers:
            Z = layer(Z)
        Z = self.avgpool(Z)
        return self.fc(Z)

    def train(self, train_generator, validation_generator, epochs=1):
        self.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator)


from load_datasets import Datasets

data = Datasets()
train_generator, validation_generator = data.load_cars_dataset()

resnet34 = ResNet34(196)
resnet34.build(input_shape=(None, 224, 224, 3))
resnet34.summary()
resnet34.compile(optimizer, loss, metrics)
resnet34.train(train_generator, validation_generator, epochs=1)

resnet34.save('resnet34.keras')