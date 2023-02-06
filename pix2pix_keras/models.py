from .utils import tf
from .losses import pix2pix_loss


class Pix2Pix(tf.keras.Model):
    """
    An implementation of Pix2Pix algorithm. Simple abstraction over keras API.
    Takes two keras Models (tf.keras.Model). First of them is generator, second one is discriminator.
    
    Current class is inherited from keras.Model.

    Attributes
    ----------
    generator : tf.keras.Model
        a model for generator part of the algorithm.
    discriminator : tf.keras.Model
        a discriminator model.
    """

    # TODO: Think about proper way of saving model

    def __init__(self, generator: tf.keras.Model, discriminator: tf.keras.Model):
        """
        Parameters
        ----------
        generator : tf.keras.Model
            Generator model (U-Net-like or another generative architectures).
        discriminator : tf.keras.Model
            Discriminative model for algorithm.
        """

        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

        # define losses and metrics
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.generator_loss_tracker = tf.keras.metrics.Mean(name="generator_loss")
        self.gan_loss_tracker = tf.keras.metrics.Mean(name="gan_loss")
        self.l1_loss_tracker = tf.keras.metrics.Mean(name="l1_loss")
        self.discriminator_loss = tf.keras.metrics.Mean(name="discriminator_loss")

        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self(x, training=True)

            disc_generated_output = self.discriminator([x, gen_output], training=True)
            disc_real_output = self.discriminator([x, y], training=True)

            loss, total_gen_loss, gan_loss, l1_loss, disc_loss = pix2pix_loss(gen_output, disc_generated_output, disc_real_output, y)
        
        generator_gradients = gen_tape.gradient(total_gen_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.generator_loss_tracker.update_state(total_gen_loss)
        self.gan_loss_tracker.update_state(gan_loss)
        self.l1_loss_tracker.update_state(l1_loss)
        self.discriminator_loss.update_state(disc_loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data

        gen_output = self(x, training=False)
        disc_generated_output = self.discriminator([x, gen_output], training=False)
        disc_real_output = self.discriminator([x, y], training=False)

        loss, total_gen_loss, gan_loss, l1_loss, disc_loss = pix2pix_loss(gen_output, disc_generated_output, disc_real_output, y)

        self.loss_tracker.update_state(loss)
        self.generator_loss_tracker.update_state(total_gen_loss)
        self.gan_loss_tracker.update_state(gan_loss)
        self.l1_loss_tracker.update_state(l1_loss)
        self.discriminator_loss.update_state(disc_loss)
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [self.loss_tracker, self.generator_loss_tracker, self.gan_loss_tracker, self.l1_loss_tracker, self.discriminator_loss]

    def call(self, inputs):
        gen_output = self.generator(inputs, training=True)
        return gen_output

