import os

import keras.utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import keras.datasets.mnist as mnist
from DenseNN import DenseNN

def model_predict(x):
    y = layer_1(x)
    y = layer_2(y)
    return y

if __name__ == '__main__':
    # Загрузка данных для обучения
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Нормирование данных
    x_train = x_train / 255
    x_test = x_test / 255

    # Вытягивание изображение в единый вектор
    x_train = tf.reshape(tf.cast(x_train, tf.float32), [-1, 28 * 28])
    x_test = tf.reshape(tf.cast(x_test, tf.float32), [-1, 28 * 28])

    # Преобразование в one hot vector
    y_train = keras.utils.to_categorical(y_train, 10)

    # Создание слоев
    # Первый слой
    layer_1 = DenseNN(128)
    # Выходной слой
    layer_2 = DenseNN(10, activate='softmax')

    #Обучение сети(нахождение весовых коэффициентов
    # с помощью градиентного спуска)
    #функция потерь
    cross_entropy = lambda y_true, y_pred: tf.reduce_mean(
        tf.losses.categorical_crossentropy(y_true, y_pred))
    #Оптимизатор для градиентого спуска
    opt = tf.optimizers.Adam(learning_rate=0.001)

    BATCH_SIZE = 32
    EPOCHS = 10
    TOTAL = x_train.shape[0]

    #Разбивка обучающей выборки на батчи
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

    for n in range(EPOCHS):
        #Суммарное значение потерь
        loss = 0
        for x_batch, y_batch in train_dataset:
            with tf.GradientTape() as tape:
                f_loss = cross_entropy(y_batch, model_predict(x_batch))

            loss += f_loss
            #Определяем градиенты
            grads = tape.gradient(f_loss, [layer_1.trainable_variables, layer_2.trainable_variables])
            #Применяем их к обучаемым параметрам первого и второго слоя
            opt.apply_gradients(zip(grads[0], layer_1.trainable_variables))
            opt.apply_gradients(zip(grads[1], layer_2.trainable_variables))

        print(loss.numpy())

    #Определение качества
    y = model_predict(x_test)
    y2 = tf.argmax(y, axis=1).numpy()
    acc = len(y_test[y_test == y2]) / y_test.shape[0] * 100
    print(acc)
