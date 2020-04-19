import numpy as np
import tensorflow as tf
from tensorflow import keras
from gan import Generator, Discriminator
from dataset import make_anime_dataset
from PIL import Image
import glob
import os



# hyper parameters
z_dim = 100
epochs = 30
batch_size = 512
learning_rate = 0.002
is_trianing = True

def save_result(val_out, val_block_size, image_path, color_mode):
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        return img

    preprocessed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])

    for b in range(val_out.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocessed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocessed[b, :, :, :]), axis = 1)

        # concat image row to final_image
        if (b + 1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis = 0)

            # reset single row
            single_row = np.array([])
    
    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis = 2)

    image = Image.fromarray(final_image)
    image.save(image_path)


def celoss_ones(logits):

    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = tf.ones_like(logits))

    return tf.reduce_mean(loss)

def celoss_zeros(logits):

    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = tf.zeros_like(logits))

    return tf.reduce_mean(loss)

def gradient_penalty(discriminator, batch_x, fake_image):

    batchsz = batch_x.shape[0]

    t = tf.random.uniform([batchsz, 1, 1, 1])
    # [b, 1, 1, 1] => [b, h, w, c]
    t = tf.broadcast_to(t, batch_x.shape)

    interplate = t * batch_x + (1 - t) * fake_image

    with tf.GradientTape() as tape:
        tape.watch([interplate])
        d_interplote_logits = discriminator(interplate, training = True)

        grads = tape.gradient(d_interplote_logits, interplate)

    # grads: [b, h, w, c] => [b, -1]
    grads = tf.reshape(grads, [grads.shape[0], -1])
    gp = tf.norm(grads, axis = 1)
    gp = tf.reduce_mean((gp-1) ** 2)

    return gp

def d_loss_fn(generator, discriminator, batch_z, batch_x, is_trianing):
    # 1. treat read image as real
    # 2. treat generated image as fake
    fake_image = generator(batch_z, is_trianing)

    d_fake_logits = discriminator(fake_image, is_trianing)
    d_read_logits = discriminator(batch_x, is_trianing)

    d_loss_real = celoss_ones(d_read_logits)
    d_loss_fake = celoss_zeros(d_fake_logits)
    gp = gradient_penalty(discriminator, batch_x, fake_image)

    loss = d_loss_fake + d_loss_real + 1. * gp

    return loss

def g_loss_fn(generator, discriminator, batch_z, is_trianing):

    fake_image = generator(batch_z, is_trianing)
    d_fake_logits = discriminator(fake_image, is_trianing)
    loss = celoss_ones(d_fake_logits)

    return loss


def main():

    img_path = glob.glob(r'./faces/*.jpg')
    dataset, img_shape, len_dataset = make_anime_dataset(img_path, batch_size)
    print('-----------')
    print(dataset, img_shape, len_dataset)
    sample = next(iter(dataset))
    print('-----------')
    print(sample.shape, tf.reduce_min(sample).numpy(), tf.reduce_max(sample).numpy())
    dataset = dataset.repeat()
    db_iter = iter(dataset)

    generator = Generator()
    generator.build(input_shape = (None, z_dim))
    discriminator = Discriminator()
    discriminator.build(input_shape = (None, 64, 64, 3))

    g_optimizer = tf.optimizers.Adam(learning_rate = learning_rate, beta_1 = 0.5)
    d_optimizer = tf.optimizers.Adam(learning_rate = learning_rate, beta_1 = 0.5)

    for epoch in range(epochs):

        batch_z = tf.random.uniform([batch_size, z_dim], minval = -1., maxval = 1.)
        batch_x = next(db_iter)

        with tf.GradientTape() as tape:
            d_loss = d_loss_fn(generator, discriminator, batch_z, batch_x, is_trianing)

            grads = tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

        with tf.GradientTape() as tape:
            g_loss = g_loss_fn(generator, discriminator, batch_z, is_trianing)

            grads = tape.gradient(g_loss, generator.trainable_variables)
            g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        
        print('-----------')
        print(epoch, 'd-loss: ', float(d_loss), 'g-loss: ', float(g_loss))

        if epoch % 10 == 0:
            z = tf.random.uniform([100, z_dim])
            fake_image = generator(z, training = False)
            img_path = os.path.join('gan_images', 'gan-%d.png' % epoch)
            save_result(fake_image.numpy(), 10, img_path, color_mode = 'P')

if __name__ == "__main__":
    main()