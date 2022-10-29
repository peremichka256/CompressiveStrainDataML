import tensorflow as tf

#Класс описывающий модель полносвязного слоя нейроной сети
class DenseNN(tf.Module):
    #Конструктор в котором определяется количество выходов
    def __init__(self, outputs, activate='relu'):
        super().__init__()
        self.outputs = outputs
        self.activate = activate
        self.fl_init = False

    #Функтор, который позволить экземпляер класса вызывать как
    # функцию принимающую x
    def __call__(self, x):
        if not self.fl_init:
            #Инициализация начальные значения весовых параметров
            self.w = tf.random.truncated_normal((x.shape[-1], self.outputs),
                                                stddev=0.1, name='w')
            self.b = tf.zeros([self.outputs], dtype=tf.float32, name='b')
            #Преобразовываем константные тензоры в переменные
            self.w = tf.Variable(self.w)
            self.b = tf.Variable(self.b)
            self.fl_init = True
        y = x @ self.w + self.b

        #Пропускание сум черех функции активации
        if self.activate == 'relu':
            return tf.nn.relu(y)
        elif self.activate == 'softmax':
            return tf.nn.softmax(y)

        return y