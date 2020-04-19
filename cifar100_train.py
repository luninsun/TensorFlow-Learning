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


conv_layers = [
    # unit 1
    layers.Conv2D(64, kernel_size = [3, 3], padding = 'same', activation = tf.nn.relu),
    layers.Conv2D(64, kernel_size = [3, 3], padding = 'same', activation = tf.nn.relu),
    layers.MaxPoll2D(pool_size = [2, 2], strides = 2, padding = 'same'),

    # unit 2
    layers.Conv2D(128, kernel_size = [3, 3], padding = 'same', activation = tf.nn.relu),
    layers.Conv2D(128, kernel_size = [3, 3], padding = 'same', activation = tf.nn.relu),
    layers.MaxPoll2D(pool_size = [2, 2], strides = 2, padding = 'same'),

    # unit 3
    layers.Conv2D(256, kernel_size = [3, 3], padding = 'same', activation = tf.nn.relu),
    layers.Conv2D(256, kernel_size = [3, 3], padding = 'same', activation = tf.nn.relu),
    layers.MaxPoll2D(pool_size = [2, 2], strides = 2, padding = 'same'),

    # unit 4
    layers.Conv2D(512, kernel_size = [3, 3], padding = 'same', activation = tf.nn.relu),
    layers.Conv2D(512, kernel_size = [3, 3], padding = 'same', activation = tf.nn.relu),
    layers.MaxPoll2D(pool_size = [2, 2], strides = 2, padding = 'same'),

    # unit 5
    layers.Conv2D(512, kernel_size = [3, 3], padding = 'same', activation = tf.nn.relu),
    layers.Conv2D(512, kernel_size = [3, 3], padding = 'same', activation = tf.nn.relu),
    layers.MaxPoll2D(pool_size = [2, 2], strides = 2, padding = 'same')
]


def main():

    conv_net = Sequential(conv_layers)

    fc_net = Sequential([
        layers.Dense(256, activation = tf.nn.relu),
        layers.Dense(128, activation = tf.nn.relu),
        layers.Dense(100, activation = tf.nn.relu)
    ])

    conv_net.build(input_shape = [None, 32, 32, 3])
    fc_net.build(input_shape = [None, 512])
    
    optimizer = optimizers.Adam(lr = 1e-4)

    variables = conv_net.trainable_variables + fc_net.trainable_variables

    for epoch in range(50):

        for step, (x, y) in enumerate(train_db):

            with tf.GradientTape() as tape:
                out = conv_net(x)

                out = tf.reshape(out, [-1, 512])

                logits = fc_net(out)

                y_onehot = tf.one_hot(y, depth = 100)

                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits = True)
                loss = tf.reduce_mean(loss)

                grads = tape.gradient(loss, variables)
                optimizer.apply_gradients(zip(grads, variables))
            
            if step % 100 == 0:
                print(epoch, step, 'loss', float(loss))


        total_num = 0
        total_correct = 0
        for x, y in test_db:
            out = conv_net(x)
            out = tf.reshape(out, [-1, 512])
            logits = fc_net(out)
            prob = tf.nn.softmax(logits, axis = 1)
            pred = tf.argmax(prob, axis = 1)
            pred = tf.cast(pred, dytpe = tf.int32)

            correct = tf.cast(tf.equal(pred, y), dtype = tf.int32)
            correct = tf.reduce_sum(correct)

            total_num += x.shape[0]
            total_correct += int(correct)

        acc = total_correct / total_num
        print(epoch, 'acc', acc)


if __name__ == '__main__':
    main()