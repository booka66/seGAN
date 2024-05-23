import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataIO import (
    load_channel_data,
    get_channel_signal_without_seizures,
    get_active_channels,
)

# Enable mixed precision training
policy = tf.keras.mixed_precision.Policy("mixed_float16")
tf.keras.mixed_precision.set_global_policy(policy)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class WasserGAN_GP:
    def __init__(self, latent_dim, max_length):
        self.latent_dim = latent_dim
        self.max_length = max_length
        self.dropout = 0.25
        self.alpha = 0.2
        self.momentum = 0.8
        self.critic_iter = 5
        self.gp_weight = 10

        self.gen_optimiser = tf.keras.optimizers.Adam(0.0002, 0.2)
        self.critic_optimiser = tf.keras.optimizers.RMSprop(0.0005)

        self.generator = self.build_generator()
        self.critic = self.build_critic()

    def build_generator(self):
        inputs = tf.keras.layers.Input(shape=(self.latent_dim,))
        x = tf.keras.layers.Dense(64)(inputs)
        x = tf.keras.layers.RepeatVector(self.max_length)(x)
        x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
        outputs = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(1, activation="tanh")
        )(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def build_critic(self):
        inputs = tf.keras.layers.Input(shape=(self.max_length, 1))
        x = tf.keras.layers.Conv1D(64, 5, strides=2, padding="same")(inputs)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Conv1D(128, 5, strides=2, padding="same")(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        outputs = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def make_fakedata(self, batch_size):
        noise = tf.random.normal((batch_size, self.latent_dim))
        return self.generator(noise, training=False), noise

    def critic_loss(self, f_logits, r_logits):
        return tf.reduce_mean(f_logits) - tf.reduce_mean(r_logits)

    def generator_loss(self, f_logits):
        return -tf.reduce_mean(f_logits)

    def gradient_penalty(self, critic, real_signal, fake_signal):
        # Cast real_signal and fake_signal to the same data type
        real_signal = tf.cast(real_signal, tf.float32)
        fake_signal = tf.cast(fake_signal, tf.float32)

        # Draw samples from a uniform distribution
        delta = tf.random.uniform([real_signal.shape[0], 1, 1], 0.0, 1.0)
        inter = real_signal + (delta * (fake_signal - real_signal))

        # Use GradientTape to watch the gradient variables.
        with tf.GradientTape() as tape:
            tape.watch(inter)
            pred = critic(inter)

        # Uses the squared difference from 1 norm as the Gradient Penalty
        grad = tape.gradient(pred, inter)[0]
        gradient_l2_norm = tf.sqrt(tf.reduce_sum(tf.square(grad)))

        return tf.reduce_mean(gradient_l2_norm)

    @tf.function
    def train_step(self, real_data):
        batch_size = tf.shape(real_data)[0]

        for _ in range(self.critic_iter):
            with tf.GradientTape() as disc_tape:
                noise = tf.random.normal([batch_size, self.latent_dim])
                gen_sig = self.generator(noise, training=True)
                f_logits = self.critic(gen_sig, training=True)
                r_logits = self.critic(real_data, training=True)

                critic_loss = self.critic_loss(f_logits, r_logits)
                gp = self.gradient_penalty(self.critic, real_data, gen_sig)
                critic_loss += self.gp_weight * tf.cast(gp, tf.float16)

            disc_grads = disc_tape.gradient(
                critic_loss, self.critic.trainable_variables
            )
            self.critic_optimiser.apply_gradients(
                zip(disc_grads, self.critic.trainable_variables)
            )

        noise = tf.random.normal([batch_size, self.latent_dim])

        with tf.GradientTape() as gen_tape:
            gen_sig = self.generator(noise, training=True)
            f_logits = self.critic(gen_sig, training=True)
            gen_loss = self.generator_loss(f_logits)

        gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimiser.apply_gradients(
            zip(gen_grads, self.generator.trainable_variables)
        )

        return {"d_loss": critic_loss, "g_loss": gen_loss}


def load_data(folder, segment_length=1000, downsample_factor=1):
    files = os.listdir(folder)[:5]
    for file in files:
        if file.endswith(".h5"):
            active_channels = get_active_channels(os.path.join(folder, file))
            for channel in active_channels:
                signal, _, _ = load_channel_data(os.path.join(folder, file), *channel)
                signal = signal[::downsample_factor]  # Downsample the signal
                for i in range(0, len(signal), segment_length):
                    segment = signal[i : i + segment_length]
                    if len(segment) < segment_length:
                        pad_width = segment_length - len(segment)
                        segment = np.pad(segment, (0, pad_width), mode="constant")
                    yield segment


# Training
data_folder = "./mv_data/"
segment_length = 50
downsample_factor = 100
data_generator = load_data(data_folder, segment_length, downsample_factor)
latent_dim = 100
epochs = 5
batch_size = 16

wgan_gp = WasserGAN_GP(latent_dim, segment_length)

for epoch in range(epochs):
    batch_data = []
    for signal in tqdm(
        data_generator, desc=f"Epoch {epoch+1}", unit="signal", leave=False
    ):
        batch_data.append(signal)
        if len(batch_data) == batch_size:
            batch_data = np.array(batch_data)
            batch_data = np.reshape(
                batch_data, (batch_size, segment_length, 1)
            )  # Reshape the data
            loss = wgan_gp.train_step(batch_data)
            batch_data = []
    if len(batch_data) > 0:
        batch_data = np.array(batch_data)
        if batch_data.shape[0] > 0:
            batch_data = np.reshape(
                batch_data, (-1, segment_length, 1)
            )  # Reshape the data
            loss = wgan_gp.train_step(batch_data)


# Save the generator model
wgan_gp.generator.save("generator_model.h5")

# Save the critic model
wgan_gp.critic.save("critic_model.h5")

# Create a new trace of n length
n = 500  # Desired length of the generated signal
generated_signal, _ = wgan_gp.make_fakedata(1)
generated_signal = tf.reshape(generated_signal, (n, 1))


# Load the saved generator model
loaded_generator = tf.keras.models.load_model("generator_model.h5")

# Load the saved discriminator model
loaded_critic = tf.keras.models.load_model("critic_model.h5")

# Generate a new signal using the loaded generator model
generated_signal, _ = wgan_gp.make_fakedata(1)
generated_signal = tf.reshape(generated_signal, (n, 1))
