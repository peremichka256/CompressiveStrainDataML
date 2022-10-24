import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import math
from tensorflow import keras
import numpy as np

#ФУНКЦИИ АВТОЗАПОЛНЕНИЯ
zero_tensor = tf.zeros((3, 3), tf.int8)
print(zero_tensor)

ones_tensor = tf.ones((4, 4),tf.int8)
print(ones_tensor)

#Единичные и нулевые матрицы по размерам
# передаваемого тензора
ones_like_tensor = tf.ones_like(zero_tensor)
print(ones_like_tensor)

zero_like_tensor = tf.zeros_like(ones_tensor)
print(zero_like_tensor)

#Тензор с единицами по диагонали
diagonal_ones = tf.eye(4)
print(diagonal_ones)

#Копия тензора с сохранением значения
copy_of_tensor = tf.identity(diagonal_ones)
print(copy_of_tensor)

#Формирование тензора с заданными значениями и размерностью
filling_tensor = tf.fill([3, 3], 14.88)
print(filling_tensor)

#Задание списка с заданными интервалами
range_tensor = tf.range(1, 10, 0.5)
print(range_tensor)

#ГЕНЕРАЦИЯ С СЛУЧАЙНЫМИ ЗНАЧЕНИЯМИ
#Нормальное распределение
normal_random = tf.random.normal((3, 3), 0, 0.1)
print(normal_random)

#Случайные значения в диапозоне
range_random = tf.random.uniform((1,10), 0, 10)
print(range_random)

#Случайные значение с seedом
tf.random.set_seed(282)
a = tf.random.normal((1, 3), 0, 0.1)
print(a)
tf.random.set_seed(282)
b = tf.random.normal((1, 3), 0, 0.1)
print(b)

#МАТЕМАТИЧЕСКИЕ ФУНКЦИИ
#Сложение тензоров
tensor_sum = tf.add(a, b)                   #Можно просто a + b
print('Сумма двух тензоров: ', tensor_sum.numpy())

#Вычитание тензоров
tensor_sub = tf.subtract(tensor_sum, b)     #Можно просто tensor_sum - b
print('Разница двух тензоров: ', tensor_sub.numpy())

#Деление тензоров поэлиментно
div_tensor = tf.divide(tensor_sum, a)       #Можно просто через tensor_sum / a
print(div_tensor.numpy())

#Умножение тезоров поэлиментно
mult_tensor = tf.multiply(tensor_sum, b)    #Можно просто через tensor_sum * b
print(mult_tensor)

#Возведение в степень
print(div_tensor**5)

#Внешнее векторное умножение
mult_tensor_external = tf.tensordot(mult_tensor, b, axes=0)
print(mult_tensor_external)

#Перемножение матрицы
first_matrix = tf.constant(tf.range(1, 10), shape=(3, 3))
second_matrix = tf.constant(tf.range(10, 19), shape=(3, 3))
mult_matrix = tf.matmul(first_matrix, second_matrix)    #first_matrix @ second_matrix
print(mult_matrix)

#Сумма элементов
element_sum = tf.reduce_sum(mult_matrix, axis=1)
print(element_sum)

#Поиск минимума и маскимума axis - по столбцам
min_item = tf.reduce_min(mult_matrix)
print('Минимальный элемент из тензора: ', min_item)
max_item = tf.reduce_max(mult_matrix)
print('Максимальный элемент из тензора: ', max_item)

#Вычисление произведения всей матрицы
mult_elements = tf.reduce_prod(first_matrix)
print(mult_elements)

#Возведение каждого элемента в квадрат
sqrt_matrix = tf.square(first_matrix)       #a ** 2
print(sqrt_matrix)

#Тригонометрические функции
cos_of_tensor = tf.cos(tf.range(-math.pi, math.pi, 1))
print(cos_of_tensor)

#Так же очень много функций в tf.keras...