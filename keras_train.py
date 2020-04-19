import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from tensorflow import keras

def preprocess(x, y):
    # [0~255] => [-1~1]
    x = 2 * tf.cast(x, dtype = tf.float32) / 255. - 1.
    y = tf.cast(y, dtype = tf.int32)

    return x, y

batchsz = 128
(x, y), (x_val, y_val) = datasets.cifar10.load_data()
y = tf.squeeze(y)   # 将input中所有维度为1的维度删除
y_val = tf.squeeze(y_val)
y = tf.one_hot(y, depth = 10)
y_val = tf.one_hot(y_val, depth = 10)
print('datasets', x.shape, y.shape, x_val.shape, y_val.shape, x.min(), x.max())

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.map(preprocess).shuffle(10000).batch(batchsz)
test_db = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_db = test_db.map(preprocess).batch(batchsz)


sample = next(iter(train_db))
print('batch:', sample[0].shape, sample[1].shape)

class MyDense(layers.Layer):
    # to replace standard layers.Dense()

    def __init__(self, input_dim, output_dim):
        super(MyDense, self).__init__()

        self.kernel = self.add_variable('w', [input_dim, output_dim])
        self.bias = self.add_variable('b', [output_dim])

    def call(self, inputs, training=None):
        x = inputs @ self.kernel + self.bias

        return x

class MyNetwork(keras.Model):

    def __init__(self):
        super(MyNetwork, self).__init__()

        self.fc1 = MyDense(32*32*3, 256)
        self.fc2 = MyDense(256, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 10)

    def call(self, inputs, training = None):

        x = tf.reshape(inputs, [-1, 32*32*3])

        x = self.fc1(x)
        x = tf.nn.relu(x)

        x = self.fc2(x)
        x = tf.nn.relu(x)

        x = self.fc3(x)
        x = tf.nn.relu(x)

        x = self.fc4(x)
        x = tf.nn.relu(x)

        x = self.fc5(x)

        return x

network = MyNetwork()
network.compile(optimizer = optimizers.Adam(lr = 1e-3),
                loss = tf.losses.CategoricalCrossentropy(from_logits = True),
                metrics = ['accuracy'])
network.fit(train_db, epochs = 15, validation_data = test_db, validation_freq = 1)

network.evaluate(test_db)

# 保存参数值
network.save_weights('ckpt/weights.ckpt')
print('saved to ckpt/weights.ckpt')

# 删除模型
del network

network = MyNetwork()
network.compile(optimizer = optimizers.Adam(lr = 1e-3),
                loss = tf.losses.CategoricalCrossentropy(from_logits = True),
                metrics = ['accuracy'])
network.load_weights('ckpt/weights.ckpt')
print('loaded weights from file')

network.evaluate(test_db)






