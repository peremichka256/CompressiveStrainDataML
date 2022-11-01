import tensorflow as tf
from DenseLayer import DenseLayer

class NeuralNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer_1 = DenseLayer(128)
        self.layer_2 = DenseLayer(10)

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = tf.nn.relu(x)
        x = self.layer_2(x)
        x = tf.nn.softmax(x)
        return x