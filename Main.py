import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import time


def function_for(s, x, y):
    for n in range(10):
        s = s + tf.matmul(s, y) * x

    return s


def function_tf(x, y):
    #При графовом представлении функция будет отброшена,
    # т.к она не влияет на вычисления
    print('Вызов функции print')
    s = tf.zeros_like(x, dtype=tf.float32)
    s = s +  tf.matmul(x, y)

    return function_for(s, x, y)


def test_function(fn):
    def wrapper(*args, **kwargs):
        stat = time.time()
        for i in range(10):
            fn(*args, **kwargs)
        dt = time.time() - stat
        print(f'Время обработки: {dt} сек')
    return wrapper


if __name__ == '__main__':
    SIZE = 1000
    x = tf.ones((SIZE, SIZE), dtype=tf.float32)
    y = tf.ones_like(x, dtype=tf.float32)

    function_tf_graph = tf.function(function_tf)

    test_function(function_tf)(x, y)
    test_function(function_tf_graph)(x, y)