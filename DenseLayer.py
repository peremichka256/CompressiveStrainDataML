import tensorflow as tf

#Класс описывающий модель полносвязного слоя нейроной сети
class DenseLayer(tf.keras.layers.Layer):
    #Units - количество нейронов в полносвязном слое
    def __init__(self, units=1):
        super().__init__()
        self.units = units

    #Вызывается для инициализации весовых коэффициентов
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer='zeros',
                                 trainable=True)

    #Не магический call вход умножаем w и прибаляем вектор весовых коэффициентов
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
