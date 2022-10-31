import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from DenseLayer import DenseLayer
import tensorflow as tf

if __name__ == '__main__':
    layer1 = DenseLayer(10)
    y = layer1(tf.constant([[1., 2., 3.]]))
    print(y)
