# 自定义模型
import tensorflow as tf
from cusLayer import MyLayer


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.l1 = MyLayer(28 * 28 * 1, 256, activation=tf.keras.activations.elu)
        self.l2 = MyLayer(256, 128, activation=tf.keras.activations.elu)
        self.l3 = MyLayer(128, 64, activation=tf.keras.activations.elu)
        self.l4 = MyLayer(64, 32, activation=tf.keras.activations.elu)
        self.l5 = MyLayer(32, 10, activation=tf.keras.activations.elu)
        self.l6 = MyLayer(10, 1, activation=tf.keras.activations.elu)

    def call(self, inputs, training=None, mask=None):
        x = self.l1(inputs)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = tf.nn.sigmoid(x)
        return x
