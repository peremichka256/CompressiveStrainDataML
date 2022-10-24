import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
import numpy as np


#Чаще всего тензоры можно понимать как константные неизменяемые
# объекты. C использованием объекта constant, где
# value - значение тензора
# dtype - тип данных тензора
# shape - размерность тензора
# name - имя тензора
#Скаляр
a = tf.constant(1, shape=(1, 1))
#Список
b = tf.constant([1, 2, 3, 4])
#Матрицы
c = tf.constant([[1, 2],
                 [2, 3],
                 [3, 4]], dtype=tf.int8)
#cast преобразует один тип данных в другой
a_float = tf.cast(a, dtype=tf.float32)
print(a)
print(b)
print(c)
print(a_float)
#Преобразование тензора в вектор numpy
b_numpy = np.array(b)
print(b_numpy)

#Задавать изменяемые тензоры можно с помощью Variable
v1 = tf.Variable(c)
v2 = tf.Variable([1, 1])
print(v1)

#Изменять тензоры можно с помощью assign_add(sub)
v2.assign([0, 0])               #Добавление
print(v2)

v3 = tf.Variable([1, 2, 3, 4, 5])
v3.assign_sub([5, 4, 3, 2, 1])  #Вычитание
print(v3)
print('Размерность тензора', v3.shape)

#Срезы в TF, но без копирования информации - это один объект
val_0 = v3[1:3]
val_0.assign(10)
print(val_0)
print(v3)

#Списочное индексирование
x = tf.constant(range(10)) + 5
#Новый тензор сформирован на основе другого
x_indx = tf.gather(x, [0, 4])
print(x, x_indx, sep='\n')

#Индексы как обращение к строке и столбцу
val_indx = c[(1)]
print(val_indx)

#Индексы как обращение к строке и столбцу
val_indx = c[(1, 1)]
print(val_indx)

#Взять все вторые элементы в каждом столбце
values_2 = c[:, 1]  #start:stop:step
print(values_2)

#Изменение размерности тензора, но новый тензор не создает
#Должна сохраняться размерность в двух тензорах
a = tf.constant(range(30))
b = tf.reshape(a, [5, 6])
reshape_tensor = tf.reshape(a, [6, -1])
print(a)
print(b.numpy())
print(reshape_tensor.numpy())

#Для транспонирования можем использовать transpose
reshape_tensor_T = tf.transpose(reshape_tensor, perm=[1, 0])
print(reshape_tensor_T.numpy())