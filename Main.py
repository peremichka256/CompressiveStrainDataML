import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras.utils


import tensorflow as tf
from keras.datasets import mnist
from DenseLayer import DenseLayer
from NeuralNetwork import NeuralNetwork


model = NeuralNetwork()
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01),
              loss=tf.losses.categorical_crossentropy,
              metrics=['accuracy'])

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

x_train = tf.reshape(tf.cast(x_train, tf.float32), [-1, 28*28])
x_test = tf.reshape(tf.cast(x_test, tf.float32), [-1, 28*28])

y_train = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)


model.fit(x_train, y_train, batch_size=32, epochs=5)
print(model.evaluate(x_test, y_test_cat))