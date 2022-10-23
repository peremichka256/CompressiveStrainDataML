import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#Создание переменных на уровне tensorflow
x = tf.Variable([[2.0]])
y = tf.Variable([[-4.0]])
#С помощью объекта GradientTape() сохряняются все
# промежуточные значения графа для последующего
# расчёта производных
with tf.GradientTape() as tape:
    #Сама функция, а значения хранятся в объекте tape
    func = (x + y) ** 2 + 2 * x * y

#используем объект tape для вычисления градиента, выдающий
# численные значения производныъ в точках 2 и 4
df = tape.gradient(func, [x,y])
print(df[0], df[1], sep='\n')

#реализация стохастического градиентного спуска
# для определения точки минимума функции
x = tf.Variable(-1.0)
#Задаём функцию через лямбда-функцию
y = lambda: x ** 2 - x

N = 100
opt = tf.optimizers.SGD(learning_rate=0.1)
for n in range (N):
    opt.minimize(y, [x])

print(x.numpy())
print(tf.__version__)

#TF нужен не только для нейронок, а для
# решения производных и оптимизации, используя графы вычисления
#TF так же пытается паралелльно переключится на GPU через CUDA
#для распаралелливания