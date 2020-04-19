import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets, Sequential
from ResNet import resnet18


def preprocess(x, y):

    x = tf.cast(x, dtype=tf.float32) / 255. - 0.5
    y = tf.cast(y, dtype=tf.int32)

    return x, y


(x, y), (x_test, y_test) = datasets.cifar100.load_data()
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)
print(x.shape, y.shape, x_test.shape, y_test.shape)

batchsz = 128

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.map(preprocess).shuffle(1000).batch(batchsz)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(batchsz)


def main():

    model = resnet18()
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()
    optimizer = optimizers.Adam(lr=1e-3)

    for epoch in range(500):

        for step, (x, y) in enumerate(train_db):

            with tf.GradientTape() as tape:

                logits = model(x, training=True)

                y_onehot = tf.one_hot(y, depth=100)

                loss = tf.losses.categorical_crossentropy(y_onehot,
                                                          logits,
                                                          from_logits=True)
                loss = tf.reduce_mean(loss)

                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads,
                                              model.trainable_variabels))

            if step % 100 == 0:
                print(epoch, step, 'loss: ', float(loss))

        total_num = 0
        total_correct = 0
        for x, y in test_db:

            logtis = model(x, training=False)
            prob = tf.nn.softmax(logtis, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_num += x.shape[0]
            total_correct += int(correct)
        acc = total_correct / total_num
        print(epoch, 'acc', acc)


if __name__ == '__main__':
    main()