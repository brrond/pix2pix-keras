from .utils import tf, discriminator_loss, generator_loss


def pix2pix_loss(y_true, y_pred):
    gen_output, disc_generated_output = y_pred
    target, disc_real_output = y_true

    gen_total_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    return gen_total_loss + disc_loss


class Pix2Pix(tf.keras.Model):
    def __init__(self, gen: tf.keras.Model, disc: tf.keras.Model):
        super().__init__()
        self.gen = gen
        self.disc = disc

    def call(self, inputs):
        gen_output = self.gen(inputs, training=True)
        disc_generated_output = self.disc([inputs, gen_output], training=True)
        return gen_output, disc_generated_output

