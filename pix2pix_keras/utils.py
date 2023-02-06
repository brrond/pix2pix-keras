import tensorflow as tf
import numpy as np


LAMBDA = 100


def create_downsample(filters: int, size, apply_batch_normalization=True):
    """Creates one block of downsampling (encoder part).

    Parameters
    ----------
    filters : int
        The number of filters for Conv2D block.
    size : int, tuple, list
        Size of the filter.
    apply_batch_normalization : bool
        Either apply batch normalization layer or not (default True).

    Returns
    -------
    tf.keras.Sequential
        a downsampling model part.
    """

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', use_bias=False))
    if apply_batch_normalization:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result


def create_upsample(filters, size, apply_dropout=False):
    """Creates one block of upsampling (decoder part).

    Parameters
    ----------
    filters : int
        The number of filters for Conv2D block.
    size : int, tuple, list
        Size of the filter.
    apply_dropout : bool
        Either apply dropout layer or not (default False). Dropout rate .5.

    Returns
    -------
    tf.keras.Sequential
        a upsampling model part.
    """
    
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result

