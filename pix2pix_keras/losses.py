from .utils import tf, LAMBDA


binary_crossentropy_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def pix2pix_loss(gen_output, disc_generated_output, disc_real_output, target):
    total_gen_loss, gan_loss, l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    total_loss = total_gen_loss + disc_loss
    return total_loss, total_gen_loss, gan_loss, l1_loss, disc_loss


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = binary_crossentropy_loss(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = binary_crossentropy_loss(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = binary_crossentropy_loss(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss