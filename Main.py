import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib.pyplot as plt
from DenseNN import DenseNN

#В машинном обучении задачу поиска решения можно
# представить в форме модели, которая должна коректно
# отобразить вход на выходы. Модель - это чаще всего нейронная сеть
model = DenseNN(1)
#print(model(tf.constant([[1.0, 2.0]]))) - неверно
#Чтобы обучить нейронную сеть нужно определить
# множество обучающей выборки
x_train = tf.random.uniform(minval=0, maxval=10, shape=(100, 2))
y_train = [a + b for a, b in x_train]

#Определение функции потерь и оптимизатор для
# градиентного спуска
loss = lambda x, y: tf.reduce_mean(tf.square(x-y))
opt = tf.optimizers.Adam(learning_rate=0.01)

#Реализация алгоритма обучения
EPOCHS = 50
for n in range(EPOCHS):
    for x, y in zip(x_train, y_train):
        x = tf.expand_dims(x, axis=0)
        y = tf.constant(y, shape=(1, 1))

        with tf.GradientTape() as tape:
            f_loss = loss(y, model(x))

        grads = tape.gradient(f_loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

    print(f_loss.numpy())

print(model.trainable_variables)
print(model(tf.constant([[1.0, 2.0]])))