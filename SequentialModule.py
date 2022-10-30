import tensorflow as tf
from DenseNN import DenseNN

class SequentialModule(tf.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = DenseNN(128)
        self.layer_2 = DenseNN(10, activate='softmax')

    def __call__(self, x):
        return self.layer_2(self.layer_1(x))