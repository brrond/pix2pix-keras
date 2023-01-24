from .utils import *


def Unet(backbone=None, input_shape=(256, 256, 3), output_channels=3):
    if backbone is None:
        inputs = tf.keras.layers.Input(shape=input_shape)

        down_stack = [
            create_downsample(64, 4, False),   # (batch_size, 128, 128, 64)
            create_downsample(128, 4),         # (batch_size, 64, 64, 128)
            create_downsample(256, 4),         # (batch_size, 32, 32, 256)
            create_downsample(512, 4),         # (batch_size, 16, 16, 512)
            create_downsample(512, 4),         # (batch_size, 8, 8, 512)
            create_downsample(512, 4),         # (batch_size, 4, 4, 512)
            create_downsample(512, 4),         # (batch_size, 2, 2, 512)
            create_downsample(512, 4),         # (batch_size, 1, 1, 512)
        ]

        up_stack = [
            create_upsample(512, 4, apply_dropout=True),    # (batch_size, 2, 2, 1024)
            create_upsample(512, 4, apply_dropout=True),    # (batch_size, 4, 4, 1024)
            create_upsample(512, 4, apply_dropout=True),    # (batch_size, 8, 8, 1024)
            create_upsample(512, 4),                        # (batch_size, 16, 16, 1024)
            create_upsample(256, 4),                        # (batch_size, 32, 32, 512)
            create_upsample(128, 4),                        # (batch_size, 64, 64, 256)
            create_upsample(64, 4),                         # (batch_size, 128, 128, 128)
        ]

        last = tf.keras.layers.Conv2DTranspose(output_channels, 4, strides=2, padding='same', activation='sigmoid')  # (batch_size, 256, 256, 3)

        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x, name='simple_unet')


