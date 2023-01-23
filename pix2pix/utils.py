import tensorflow as tf

LAMBDA = 100
binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def create_downsample(filters, size, apply_batch_normalization=True):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', use_bias=False))
    if apply_batch_normalization:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result


def create_upsample(filters, size, apply_dropout=False):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = binary_crossentropy(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = binary_crossentropy(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = binary_crossentropy(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss