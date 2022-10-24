import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import math
from tensorflow import keras
import numpy as np

#Реализация автоматического дифференцирования
w = tf.Variable(tf.random.normal((3, 2)))
b = tf.Variable(tf.zeros(2, dtype=tf.float32))
x = tf.Variable([[-2.0, 1.0, 3.0]])
#В менеджере контекста записываем все промежуточные вычисления
# в tape
with tf.GradientTape() as tape:
    y = x @ w + b
    loss = tf.reduce_mean(y ** 2)

#Обратный проход по графу вычислений
df = tape.gradient(loss, [w, b])
print(df[0], df[1], sep='\n')