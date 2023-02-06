from .utils import tf, create_downsample


def PatchGAN(input_shape=[256, 256, 3]):
    """Generates simple PatchGAN model as tf.keras.Model.

    Parameters
    ----------
    input_shape : list
        The list of shapes of input images.

    Returns
    -------
    tf.keras.Model
        a keras Model that defines PatchGAN discriminator.
    """

    inp = tf.keras.layers.Input(shape=input_shape, name='input_image')
    tar = tf.keras.layers.Input(shape=input_shape, name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

    down1 = create_downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
    down2 = create_downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
    down3 = create_downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1, use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1)(zero_pad2)  # (batch_size, 30, 30, 1)
    return tf.keras.Model(inputs=[inp, tar], outputs=last, name='simple_patchGAN')