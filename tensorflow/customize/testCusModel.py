# 测试自定义模型
import tensorflow as tf
from cusModel import MyModel

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
shape = tf.shape(x_train)
x_train = tf.reshape(x_train, shape=(shape[0], shape[1] * shape[2]))
x_train = tf.cast(x_train, tf.float32)
y_train = tf.expand_dims(y_train, axis=-1)
print(tf.shape(x_train))
print(tf.shape(y_train))

model = MyModel()

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=["accuracy"])

print("--------------------------")
model.build(input_shape=(None, 28 * 28 * 1))
print(model.summary())

print("--------------------------")
noise = tf.random.normal(shape=(1, 28 * 28 * 1))
result = model(noise)
print(tf.shape(result)) # [1,1]

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(64)
model.fit(x_train, y_train,epochs=5)
