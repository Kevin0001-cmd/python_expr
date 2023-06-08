# 自定义层
import tensorflow as tf


class MyLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, activation=None):
        super(MyLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight(name="kernel", shape=[self.input_dim, self.output_dim]
                                      , initializer='glorot_normal')
        self.bias = self.add_weight(name="bias", shape=[self.output_dim]
                                    , initializer='zeros')
        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        output = self.activation(inputs @ self.kernel + self.bias)
        return output
