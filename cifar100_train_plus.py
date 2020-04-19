import tensorflow as tf
from tensorflow.keras import layers, Sequential, optimizers, datasets

(x, y), (x_test, y_test) = datasets.cifar100.load_data()
y = tf.queeze(y, axis = 1)
y_test = tf.queeze(y_test, axis = 1)
print(x.shape, y.shape, x_test.shape, y_test.shape)

def preprocess(x, y):

    x = tf.cast(x, dtype = tf.float32) / 255.
    y = tf.cast(y, dtype = tf.int32)

    return x, y

batchsz = 128

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.map(preprocess).shuffle(1000).batch(batchsz)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(batchsz)


class MyDense(layers.Layer):

    def __init__(self, kernel_size, channel):
        super(MyDense, self).__init__()

        self.kernel_size = kernel_size
        self.channel = channel

    def call():
        pass