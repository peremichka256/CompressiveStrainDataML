#Однако проблема в застревании при локальных минимумах,
#что не позволяет достигнуть лучшего решения
#Для оптимизации поисков в больших массивах данных, данные
# делятся на сегмент(мини-батчи) - стохастичесуий градиентый спуск:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib.pyplot as plt

TOTAL_POINTS = 1000

x = tf.random.uniform(shape=[TOTAL_POINTS], minval=0, maxval=10)
noise = tf.random.normal(shape=[TOTAL_POINTS], stddev=0.2)

k_true = 0.7
b_true = 2.0

y = x * k_true + b_true + noise

k = tf.Variable(0.0)
b = tf.Variable(0.0)

#Можно уменьшать количество EPOCH, за счёт чего уменьшается время вычисления
EPOCHS = 50
learning_rate = 0.02
BATCH_SIZE = 100
num_steps = TOTAL_POINTS // BATCH_SIZE

#Для избежания застревания в локальных минимумах можно использовать:
#метод моментов tf.optimizers.SGD(momentum=0.5, learning_rate=0.2)
#метод Нестерова tf.optimizers.SGD(momentum=0.5, nesterow=Trues, learning_rate=0.2)
#Adagrad tf.optimizers.Adagrad(learning_rate=0.2), но шаг обучения уменьшается
#Adadelta tf.optimizers.Adadelta(learning_rate=1.0)
#RMSProp tf.optimizers.RMSprop(learning_rate=0.01)
#Adam tf.optimizers.Adam(learning_rate=0.01)
opt = tf.optimizers.SGD(learning_rate=0.02)

for n in range(EPOCHS):
    for n_batch in range(num_steps):
        #Но разбивку на сегменты нужно делать самому
        y_batch = y[n_batch * BATCH_SIZE : (n_batch + 1) * BATCH_SIZE]
        x_batch = x[n_batch * BATCH_SIZE : (n_batch + 1) * BATCH_SIZE]

        with tf.GradientTape() as t:
            f = k * x + b
            loss = tf.reduce_mean(tf.square(y - f))

        dk, db = t.gradient(loss, [k, b])
        opt.apply_gradients(zip([dk, db], [k, b]))
        #k.assign_sub(learning_rate * dk)
        #b.assign_sub(learning_rate * db)

print(k, b, sep='\n')
y_pr = k * x + b
plt.scatter(x, y, s=2)
plt.scatter(x, y_pr, c='r', s=2)
plt.show()

