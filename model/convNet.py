import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras as tfk
tfkb = tfk.backend
tfkl = tfk.layers


class ConvNet:  # Convolutional Net
    def __init__(self, input_shape=(), epochs=100, learning_rate=0.001, batch_size=50, padding='same',
                 kernel_size=None, stride=2):
        if kernel_size is None:
            self.kernel_size = [3, 3]
        else:
            self.kernel_size = kernel_size
        self.epochs = epochs
        self.input_shape = input_shape
        print(self.input_shape)
        self.stride = stride
        self.filters = 4  # The base filter number
        self.out_activ_func = 'sigmoid'
        self.activ_func = 'relu'
        self.padding = padding  # SAME or VALID
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.logs = None

        self.model = None
        # Network structure
        self.x = None
        #self.images = tfk.Input(dtype=tf.float32, shape=input_shape, name="Images")
        self.structure_net()

        optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

        # Print model
        self.model.summary()


    def structure_net(self):

        self.model = tfk.models.Sequential([
            tfkl.Conv2D(self.filters, self.kernel_size, self.stride,  padding='same',
                        activation=self.activ_func, input_shape=self.input_shape, data_format='channels_last'),
            tfkl.MaxPool2D((2, 2), data_format='channels_last'),
            tfkl.Conv2D(self.filters * 2, self.kernel_size, padding=self.padding, activation=self.activ_func,
                        data_format='channels_last'),
            tfkl.MaxPool2D((2, 2), data_format='channels_last'),
            tfkl.Conv2D(self.filters * 4, self.kernel_size, padding=self.padding, activation=self.activ_func,
                        data_format='channels_last'),
            tfkl.MaxPool2D((2, 2)),
            tfkl.Flatten(),
            tfkl.Dense(1000, activation=self.activ_func),
            tfkl.Dense(1, activation=self.out_activ_func)
        ])


    def fit(self, x, y):
        self.logs = self.model.fit(x, y, batch_size=self.batch_size, epochs=self.epochs)


    def predict(self, x):
        return self.model.predict(x)

    def visualize_perf(self):
        print(self.logs.history.keys())
        plt.plot(self.logs.history['acc'], label='accuracy')
        plt.plot(self.logs.history['loss'], label='loss')
        plt.legend()
        plt.show()
